"""
生成模块与提示词工程 第二部分 - 医学问答生成流水线（MedicalGenerationPipeline）

把前面做好的三个组件串起来：
    上下文组装器  14_context_assembler.py   ContextAssembler
    医学提示词模板 15_prompt_templates.py    PROMPT_STAGES（四个阶段）
    本地LLM       19_llm_generator.py        LLMGenerator

输入是用户的问题 + 检索重排序之后的参考文档列表（13_retrieval_pipeline.py的输出），
输出是最终答案、参考文献、中间结果和各种统计信息。流程分六步：

    1. 上下文组装        去重、选择、拼接、按token预算截断
    2. 检索结果评估(可选) 让模型逐个文档判断"相关吗"，输出JSON，据此把不相关的文档从上下文里筛掉
    3. 生成答案草稿
    4. 批判性审查(可选)   检查草稿有没有编造、夸大、引用错误
    5. 生成最终答案       审查成功就把审查意见交给最终组装器修订；没有审查就直接用草稿
    6. 后处理            把[来源: 标题]换成[1][2]编号、整理格式、附上参考文献和免责声明

任何一个可选阶段失败，都会退回到"跳过这个阶段"继续往下走，不会让整个流程中断；
只有生成草稿这一步失败，才会返回失败信息。另外，如果上下文组装出来是空的，不会调用模型，
直接返回NO_EVIDENCE_ANSWER（"没有检索到相关的文献证据……"）。

用法：
    作为库用：pipeline = MedicalGenerationPipeline(); result = pipeline.generate(问题, 文档列表);
              print(format_result_report(result))。generate()返回的字典有这些字段：
              query、answer、context_metadata、generation_metrics、intermediate_results、sources、timestamp
    直接运行：python 20_medical_generation_pipeline.py
              没有命令行参数，用文件底部的3条内置示例文档（DEMO_DOCS）和一个固定问题跑一遍完整流程，
              把format_result_report的结果打印出来。只需要启动Ollama，不加载检索模型，也不写文件；
              连不上Ollama就打印原因然后以状态码1退出。
更完整的测试看 21_test_generation_pipeline.py（真实检索）和 22_verify_generation_logic.py（离线假模型）。
"""

import importlib
import difflib
import re
import threading
import time

_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
_m_llm_generator = importlib.import_module("19_llm_generator")
LLMGenerator = _m_llm_generator.LLMGenerator
_m_prompt_templates = importlib.import_module("15_prompt_templates")
PROMPT_STAGES = _m_prompt_templates.PROMPT_STAGES

DISCLAIMER = "本回答基于检索到的文献证据整理生成，不构成医疗建议，具体诊疗请遵医嘱。"
DISCLAIMER_KEYWORD = "不构成医疗建议"
NO_EVIDENCE_ANSWER = "没有检索到相关的文献证据，无法回答这个问题。"

# 阶段名字，stage_times/stage_success里的key就是这几个
STAGE_NAMES = [
    "context_assembly", "evidence_evaluation", "answer_generation",
    "critical_review", "final_assembly", "post_processing",
]

# 上下文里每个chunk的标题行，格式是14_context_assembler.py拼出来的 "[来源: 文献标题]"
TITLE_LINE_PATTERN = re.compile(r"^\[来源: (.*)\]\s*$")
# 模型答案里的引用标记（冒号可能是中文的）
CITATION_PATTERN = re.compile(r"\[来源[:：]\s*(.+?)\s*\]")

# 标题模糊匹配的阈值：模型引用文献标题的时候可能漏字、改字，相似度达到这个值就认为是同一篇
TITLE_MATCH_RATIO = 0.85

# 让证据评估器输出的JSON结构（会加在system prompt末尾，见LLMGenerator._json_instruction）
EVALUATION_JSON_FORMAT = {
    "overall": "充分 / 部分充分 / 不充分",
    "reason": "一句话说明总体判断的理由",
    "documents": [
        {"id": 1, "relevant": True, "strength": "强 / 中 / 弱", "comment": "一句话评价这个文档"}
    ],
    "gaps": "现有证据还缺什么，没有缺口写无",
}

# 补充在评估提示词末尾的说明：告诉模型文档编号怎么对应、relevant怎么填
EVALUATION_EXTRA_NOTE = (
    "\n\n补充说明：每个证据片段前面有【文档N】的编号。输出JSON里的documents要把每一个文档都评估一条，"
    "id填这个编号；relevant只能填true或false：片段确实在讨论问题涉及的具体药物/疾病/机制才填true，"
    "只是主题沾边或者根本没提到的填false。"
)

RELEVANT_TRUE_WORDS = {"true", "yes", "是", "相关"}
RELEVANT_FALSE_WORDS = {"false", "no", "否", "不相关"}


class MedicalGenerationPipeline:
    """医学问答生成流水线：上下文组装 -> 证据评估(可选) -> 答案草稿 -> 批判性审查(可选) -> 最终答案 -> 后处理"""

    def __init__(self, assembler=None, llm=None, prompt_stages=None, max_context_tokens=2000,
                 enable_evaluation=True, enable_review=True):
        """
        Args:
            assembler: 上下文组装器，不传就新建一个（会加载tokenizer）
            llm: LLMGenerator，不传就新建一个（会测试Ollama连接）
            prompt_stages: 提示词模板字典，不传用15_prompt_templates.py里的PROMPT_STAGES
            max_context_tokens: 上下文的token预算
            enable_evaluation / enable_review: 默认要不要做"检索结果评估"和"批判性审查"，
                                               generate()的时候可以单独覆盖
        """
        # 初始化组件
        self.assembler = assembler if assembler is not None else ContextAssembler(max_context_tokens=max_context_tokens)
        self.llm = llm if llm is not None else LLMGenerator()
        self.prompt_stages = prompt_stages if prompt_stages is not None else PROMPT_STAGES
        self.max_context_tokens = max_context_tokens
        self.enable_evaluation = enable_evaluation
        self.enable_review = enable_review
        # 批量并行生成的时候，多个线程会同时用同一个assembler（里面的tokenizer不一定线程安全），所以组装这一步加锁
        self._assemble_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def generate(self, query, retrieved_docs, enable_evaluation=None, enable_review=None,
                 max_context_tokens=None, context_result=None):
        """
        完整的生成流程。

        Args:
            query: 用户的问题
            retrieved_docs: 检索+重排序之后的参考文档字典列表
            context_result: 可选。已经组装好的上下文（assembler.assemble的返回值）。
                            缓存模块要先组装上下文才能算缓存键，传进来可以避免重复组装；不传就在这里组装
        Returns:
            结果字典，字段见 _build_result
        """
        if enable_evaluation is None:
            enable_evaluation = self.enable_evaluation
        if enable_review is None:
            enable_review = self.enable_review
        if max_context_tokens is None:
            max_context_tokens = self.max_context_tokens

        start_total = time.time()

        # run里记录这一次生成过程中的各种统计，最后放进结果字典
        run = {
            "stage_times": {},
            "stage_success": {name: None for name in STAGE_NAMES},   # None表示这个阶段被跳过了
            "token_counts": {},
            "warnings": [],
        }
        intermediate = {
            "evidence_evaluation": None, "draft_answer": None, "review_feedback": None,
            "context_filter": None, "final_answer_raw": None, "citation_check": None,
            "context_text": None, "context_text_filtered": None,    # 给评估器/日志用：组装后的上下文，和筛选后真正给模型看的上下文
        }

        # ---- 1. 上下文组装 ----
        t = time.time()
        if context_result is None:
            with self._assemble_lock:
                context_result = self.assembler.assemble(retrieved_docs, max_context_tokens=max_context_tokens)
        intermediate["context_text"] = context_result["context_text"]
        run["stage_times"]["context_assembly"] = time.time() - t
        run["stage_success"]["context_assembly"] = bool(context_result["context_text"].strip())

        if not run["stage_success"]["context_assembly"]:
            # 没有任何可用的上下文，不用调用模型，直接说明没有证据
            run["warnings"].append("上下文为空，没有调用模型")
            return self._build_result(query, NO_EVIDENCE_ANSWER, context_result, [], run, intermediate,
                                      start_total)

        blocks = self._split_context_blocks(context_result["context_text"], context_result["included_chunks"])

        # ---- 2. 检索结果评估（可选）+ 用评估结果筛选上下文 ----
        kept_blocks = blocks
        evaluation_text = "（未进行证据评估）"
        if enable_evaluation:
            kept_blocks, evaluation_text = self._evaluate_and_filter(query, blocks, run, intermediate)

        # 后面的草稿、审查都用筛选之后的上下文；kept_chunks对应留下来的文档块，最后列参考文献也只列它们
        filtered_context = "\n\n".join(b["raw"] for b in kept_blocks)
        kept_chunks = [b["chunk"] for b in kept_blocks if b["chunk"] is not None]
        intermediate["context_text_filtered"] = filtered_context

        # ---- 3. 提取评估结果，生成答案草稿 ----
        stage = self.prompt_stages["answer_generator"]
        user_prompt = stage.user_prompt_template.format(
            question=query, context=filtered_context, evidence_evaluation=evaluation_text)
        draft_result = self._call_stage("answer_generator", "answer_generation", user_prompt, run)
        if not draft_result["success"]:
            message = f"答案生成失败：{draft_result['error']}"
            return self._build_result(query, message, context_result, self._format_sources(kept_chunks),
                                      run, intermediate, start_total)
        draft_answer = draft_result["text"]
        intermediate["draft_answer"] = draft_answer

        # ---- 4. 批判性审查（可选）----
        review_feedback = None
        if enable_review:
            stage = self.prompt_stages["critical_reviewer"]
            user_prompt = stage.user_prompt_template.format(
                question=query, context=filtered_context, draft_answer=draft_answer)
            review_result = self._call_stage("critical_reviewer", "critical_review", user_prompt, run)
            if review_result["success"]:
                review_feedback = review_result["text"]
                intermediate["review_feedback"] = review_feedback

        # ---- 5. 生成最终答案 ----
        if review_feedback is not None:
            # 审查成功：把草稿和审查意见交给最终组装器
            stage = self.prompt_stages["final_assembler"]
            user_prompt = stage.user_prompt_template.format(
                question=query, draft_answer=draft_answer, review_feedback=review_feedback)
            final_result = self._call_stage("final_assembler", "final_assembly", user_prompt, run)
            # 最终组装失败的话也退回草稿，不让整个流程失败
            final_answer = final_result["text"] if final_result["success"] else draft_answer
        else:
            # 没有审查（关闭了或者失败了）：直接用草稿
            final_answer = draft_answer
        intermediate["final_answer_raw"] = final_answer

        # ---- 6. 后处理 ----
        t = time.time()
        sources = self._format_sources(kept_chunks)
        answer, citation_check = self._postprocess(final_answer, sources)
        intermediate["citation_check"] = citation_check
        run["stage_times"]["post_processing"] = time.time() - t
        run["stage_success"]["post_processing"] = True
        for title in citation_check["unmatched_citations"]:
            run["warnings"].append(f"答案里引用了对不上的文献：{title}")

        return self._build_result(query, answer, context_result, sources, run, intermediate, start_total)

    # ------------------------------------------------------------------
    # 调用模型的统一入口
    # ------------------------------------------------------------------
    def _call_stage(self, stage_key, metric_name, user_prompt, run, json_format=None):
        # 用某个阶段的模板设置（system prompt、temperature、max_tokens）调用一次模型，
        # 并把耗时、token数、成功与否记到run里
        stage = self.prompt_stages[stage_key]
        t = time.time()
        result = self.llm.generate(
            user_prompt,
            system_prompt=stage.system_prompt,
            temperature=stage.temperature,
            max_tokens=stage.max_tokens,
            json_format=json_format,
        )
        run["stage_times"][metric_name] = time.time() - t
        run["token_counts"][metric_name] = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        }
        run["stage_success"][metric_name] = result["success"]
        if not result["success"]:
            run["warnings"].append(f"{metric_name} 失败：{result['error']}")
        elif result["truncated"]:
            run["warnings"].append(f"{metric_name} 的输出到了max_tokens={stage.max_tokens}被截断")
        return result

    # ------------------------------------------------------------------
    # 第2步：评估 + 筛选
    # ------------------------------------------------------------------
    def _split_context_blocks(self, context_text, included_chunks):
        # 检测上下文里的文档标题行 "[来源: ...]"，把上下文切成一个个文档块，编号从1开始。
        # 标题行必须在开头，或者前面是空行（chunk之间用空行隔开），避免把正文里碰巧长得像标题的行误判了
        lines = context_text.split("\n")
        blocks = []
        current = None
        for i, line in enumerate(lines):
            match = TITLE_LINE_PATTERN.match(line)
            if match and (i == 0 or lines[i - 1] == ""):
                current = {"title": match.group(1), "lines": [line]}
                blocks.append(current)
            elif current is not None:
                current["lines"].append(line)

        result = []
        same_count = len(blocks) == len(included_chunks)
        for index, block in enumerate(blocks):
            chunk = None
            if same_count:
                chunk = included_chunks[index]
            else:
                # 数量对不上（比如正文里有长得像标题的行），退回按标题找
                for c in included_chunks:
                    if c.source == block["title"]:
                        chunk = c
                        break
            result.append({
                "id": index + 1,
                "title": block["title"],
                "raw": "\n".join(block["lines"]).strip(),
                "chunk": chunk,
            })
        return result

    def _evaluate_and_filter(self, query, blocks, run, intermediate):
        # 给每个文档块加上【文档N】的编号交给证据评估器，要求它输出JSON，
        # 再根据JSON里每个文档的relevant把不相关的文档块去掉。
        # 返回(留下的文档块, 传给答案生成器的评估文字)
        numbered_context = "\n\n".join(f"【文档{b['id']}】{b['raw']}" for b in blocks)
        stage = self.prompt_stages["evidence_evaluator"]
        user_prompt = stage.user_prompt_template.format(question=query, context=numbered_context)
        user_prompt += EVALUATION_EXTRA_NOTE

        result = self._call_stage("evidence_evaluator", "evidence_evaluation", user_prompt, run,
                                  json_format=EVALUATION_JSON_FORMAT)
        if not result["success"]:
            return blocks, "（证据评估失败，未能评估）"

        parsed = self._normalize_evaluation(result["json"]) if result["json_ok"] else None
        intermediate["evidence_evaluation"] = {"raw_text": result["text"], "parsed": parsed}

        if parsed is None:
            # 模型回复了，但JSON没提取出来：不筛选，评估文字直接用模型的原话
            run["stage_success"]["evidence_evaluation"] = False
            run["warnings"].append("evidence_evaluation 的回复里没有提取出有效的JSON，没有筛选上下文")
            return blocks, result["text"]

        kept_blocks, filter_info = self._filter_blocks(blocks, parsed)
        intermediate["context_filter"] = filter_info
        if filter_info["fallback_all_irrelevant"]:
            run["warnings"].append("评估器把所有文档都判为不相关，为避免上下文为空，保留了全部文档")
        return kept_blocks, self._format_evaluation_text(parsed, kept_blocks)

    def _to_int(self, value):
        # 文档编号可能是 1、"1"、"文档1"，都转成整数；转不了返回None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        digits = re.findall(r"\d+", str(value))
        if digits:
            return int(digits[0])
        return None

    def _to_relevant(self, value):
        # relevant可能是true/false，也可能是"是"/"否"/"true"之类的字符串；认不出来返回None
        if isinstance(value, bool):
            return value
        word = str(value).strip().lower()
        if word in RELEVANT_TRUE_WORDS:
            return True
        if word in RELEVANT_FALSE_WORDS:
            return False
        return None

    def _normalize_evaluation(self, data):
        # 检查并整理评估器输出的JSON。格式不对返回None
        if not isinstance(data, dict):
            return None
        documents = []
        raw_documents = data.get("documents")
        if isinstance(raw_documents, list):
            for item in raw_documents:
                if not isinstance(item, dict):
                    continue
                doc_id = self._to_int(item.get("id"))
                if doc_id is None:
                    continue
                documents.append({
                    "id": doc_id,
                    "relevant": self._to_relevant(item.get("relevant")),
                    "strength": str(item.get("strength", "")).strip(),
                    "comment": str(item.get("comment", "")).strip(),
                })
        # documents和overall一个都没有，说明这个JSON不是评估结果
        if not documents and "overall" not in data:
            return None
        return {
            "overall": str(data.get("overall", "")).strip(),
            "reason": str(data.get("reason", "")).strip(),
            "gaps": str(data.get("gaps", "")).strip(),
            "documents": documents,
        }

    def _filter_blocks(self, blocks, parsed):
        # 基于文档id筛选：只有评估器明确写了relevant=false的文档才去掉，
        # 没提到的、判断不出来的都保留（小模型的判断不一定可靠，宁可多留不要误删）
        irrelevant = {}
        for doc in parsed["documents"]:
            if doc["relevant"] is False:
                irrelevant[doc["id"]] = doc["comment"]

        kept = []
        removed = []
        for block in blocks:
            if block["id"] in irrelevant:
                removed.append({"id": block["id"], "title": block["title"], "comment": irrelevant[block["id"]]})
            else:
                kept.append(block)

        fallback = False
        if not kept:
            kept = blocks
            removed = []
            fallback = True

        info = {
            "total_documents": len(blocks),
            "kept_ids": [b["id"] for b in kept],
            "removed": removed,
            "fallback_all_irrelevant": fallback,
        }
        return kept, info

    def _format_evaluation_text(self, parsed, kept_blocks):
        # 把评估结果整理成一段文字，作为答案生成器提示词里的"证据评估结果"。
        # 只写留下来的文档，并且用文献标题而不是编号指代（生成阶段的上下文里没有编号）
        lines = []
        if parsed["overall"]:
            overall = f"总体判断：{parsed['overall']}"
            if parsed["reason"]:
                overall += f"（{parsed['reason']}）"
            lines.append(overall)

        comments = {}
        for doc in parsed["documents"]:
            comments[doc["id"]] = doc
        for block in kept_blocks:
            doc = comments.get(block["id"])
            if doc is None:
                continue
            detail = []
            if doc["strength"]:
                detail.append(f"证据强度{doc['strength']}")
            if doc["comment"]:
                detail.append(doc["comment"])
            if detail:
                lines.append(f"- {block['title']}：" + "；".join(detail))

        if parsed["gaps"] and parsed["gaps"] != "无":
            lines.append(f"证据缺口：{parsed['gaps']}")
        if not lines:
            return "（证据评估没有给出有效内容）"
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 第6步：后处理
    # ------------------------------------------------------------------
    def _format_sources(self, chunks):
        """
        整理参考来源。同一篇文献的多个chunk合成一条，按第一次出现的顺序编号（ref_id从1开始）。
        """
        sources = []
        by_title = {}
        for chunk in chunks:
            meta = chunk.metadata
            section = meta.get("section_title")
            if chunk.source in by_title:
                entry = by_title[chunk.source]
                entry["chunk_ids"].append(chunk.chunk_id)
                if section and section not in entry["sections"]:
                    entry["sections"].append(section)
                if chunk.relevance_score > entry["best_relevance_score"]:
                    entry["best_relevance_score"] = chunk.relevance_score
            else:
                entry = {
                    "ref_id": len(sources) + 1,
                    "title": chunk.source,
                    "doc_id": meta.get("doc_id"),
                    "journal": meta.get("journal"),
                    "pub_year": meta.get("pub_year"),
                    "chunk_ids": [chunk.chunk_id],
                    "sections": [section] if section else [],
                    "best_relevance_score": chunk.relevance_score,
                    "cited": False,        # 答案里有没有引用它，后处理时更新
                }
                by_title[chunk.source] = entry
                sources.append(entry)
        return sources

    def _normalize_title(self, title):
        # 比较标题时忽略大小写、空格和标点
        return re.sub(r"[\W_]+", "", title.lower())

    def _match_title(self, cited, sources):
        # 在已知来源里找模型引用的标题。依次试：规范化后完全相同 -> 一方包含另一方 -> 模糊相似度。
        # 找到返回ref_id，找不到返回None
        cited_norm = self._normalize_title(cited)
        if not cited_norm:
            return None

        for s in sources:
            if self._normalize_title(s["title"]) == cited_norm:
                return s["ref_id"]

        if len(cited_norm) >= 15:
            for s in sources:
                title_norm = self._normalize_title(s["title"])
                if cited_norm in title_norm or title_norm in cited_norm:
                    return s["ref_id"]

        best_ratio = 0.0
        best_ref = None
        for s in sources:
            ratio = difflib.SequenceMatcher(None, cited_norm, self._normalize_title(s["title"])).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_ref = s["ref_id"]
        if best_ratio >= TITLE_MATCH_RATIO:
            return best_ref
        return None

    def _add_citation_markers(self, text, sources):
        # 把答案里的 [来源: 文献标题] 换成 [编号]。返回(新文本, 对不上的引用列表)
        # 先按已知标题做精确替换（标题里自己带方括号的时候，正则会匹配错，精确替换不受影响）
        for s in sources:
            for colon in (":", "："):
                text = text.replace(f"[来源{colon} {s['title']}]", f"[{s['ref_id']}]")
                text = text.replace(f"[来源{colon}{s['title']}]", f"[{s['ref_id']}]")

        unmatched = []

        def replace_one(match):
            cited = match.group(1)
            ref_id = self._match_title(cited, sources)
            if ref_id is None:
                unmatched.append(cited)
                return match.group(0)
            return f"[{ref_id}]"

        text = CITATION_PATTERN.sub(replace_one, text)
        return text, unmatched

    def _beautify(self, text):
        # 整理格式：统一换行符、统一列表符号、去掉markdown加粗，#标题改成【标题】，压缩多余空行
        # （<think>思考内容在19_llm_generator.py的_clean_text里就去掉了，这里不处理）
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for line in text.split("\n"):
            line = line.rstrip()
            line = line.replace("**", "")
            heading = re.match(r"^#{1,6}\s*(.+)$", line)
            if heading:
                line = f"【{heading.group(1).strip()}】"
            line = re.sub(r"^(\s*)[*•·]\s+", r"\1- ", line)
            lines.append(line)
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _remove_model_disclaimer(self, text):
        # 模型自己按提示词要求写的免责声明去掉，统一在最后加标准的那一句，避免重复或者措辞不一样
        # 正则的意思：把含有"不构成医疗建议"的那一句（从上一个句号/换行之后，到下一个句号为止）整句删掉
        text = re.sub(r"[^。\n]*" + DISCLAIMER_KEYWORD + r"[^。\n]*。?", "", text)
        return text.strip()

    def _postprocess(self, answer, sources):
        """添加引用标记 -> 美化格式 -> 附上参考文献 -> 加免责声明。返回(最终答案文本, 引用检查结果)"""
        body, unmatched = self._add_citation_markers(answer, sources)
        body = self._remove_model_disclaimer(body)
        body = self._beautify(body)

        # 哪些来源在答案里被引用了（此时正文里的[来源: 标题]已经换成[编号]，参考文献还没拼上去，所以只会数到正文里的编号）
        cited_refs = set(int(n) for n in re.findall(r"\[(\d+)\]", body))
        for s in sources:
            s["cited"] = s["ref_id"] in cited_refs

        parts = [body]
        cited_sources = [s for s in sources if s["cited"]]
        if cited_sources:
            parts.append("参考文献：\n" + "\n".join(self._format_reference_line(s) for s in cited_sources))
        elif sources:
            parts.append("检索到的相关文献（回答中没有标注具体引用）：\n"
                         + "\n".join(self._format_reference_line(s) for s in sources))
        parts.append("免责声明：" + DISCLAIMER)

        citation_check = {
            "cited_ref_ids": sorted(cited_refs),
            "unmatched_citations": unmatched,
            "has_citation": bool(cited_refs),
        }
        return "\n\n".join(parts), citation_check

    def _format_reference_line(self, source):
        """把一条来源写成参考文献的一行：[编号] 标题（期刊，年份），没有期刊/年份就不写括号"""
        line = f"[{source['ref_id']}] {source['title']}"
        extra = []
        if source.get("journal"):
            extra.append(str(source["journal"]))
        if source.get("pub_year"):
            extra.append(str(source["pub_year"]))
        if extra:
            line += "（" + "，".join(extra) + "）"
        return line

    # ------------------------------------------------------------------
    # 组装最终结果
    # ------------------------------------------------------------------
    def _build_result(self, query, answer, context_result, sources, run, intermediate, start_total):
        # 计算总时间，并按约定的结构组装结果
        total_time = time.time() - start_total

        total_prompt = 0
        total_completion = 0
        for counts in run["token_counts"].values():
            total_prompt += counts["prompt_tokens"]
            total_completion += counts["completion_tokens"]
        token_counts = dict(run["token_counts"])
        token_counts["total_prompt_tokens"] = total_prompt
        token_counts["total_completion_tokens"] = total_completion

        return {
            "query": query,
            "answer": answer,
            "context_metadata": context_result["metadata"],
            "generation_metrics": {
                "total_time_seconds": total_time,
                "stage_times": run["stage_times"],
                "token_counts": token_counts,
                "stage_success": run["stage_success"],
                "answer_length_chars": len(answer),
                "warnings": run["warnings"],
            },
            "intermediate_results": intermediate,
            "sources": sources,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


def format_result_report(result, show_intermediate=True):
    """把一次生成的结果整理成方便阅读的文字：问题、答案、参考来源、各项统计"""
    metrics = result["generation_metrics"]
    meta = result["context_metadata"]
    lines = []
    lines.append("=" * 60)
    lines.append(f"问题：{result['query']}")
    lines.append(f"时间：{result['timestamp']}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("【最终答案】")
    lines.append(result["answer"])
    lines.append("")

    lines.append("【参考来源】")
    if result["sources"]:
        for s in result["sources"]:
            mark = "已引用" if s["cited"] else "未引用"
            lines.append(f"  [{s['ref_id']}] {s['title']}")
            lines.append(f"      {mark}  章节={s['sections']}  期刊={s['journal']}  年份={s['pub_year']}"
                         f"  最高相关性={s['best_relevance_score']:.3f}  chunk={s['chunk_ids']}")
    else:
        lines.append("  （无）")
    lines.append("")

    lines.append("【生成统计】")
    lines.append(f"  总耗时：{metrics['total_time_seconds']:.1f}秒    答案长度：{metrics['answer_length_chars']}个字符")
    lines.append("  各阶段（耗时 / 状态 / 输入token / 输出token）：")
    for name in STAGE_NAMES:
        success = metrics["stage_success"][name]
        status = "跳过" if success is None else ("成功" if success else "失败")
        seconds = metrics["stage_times"].get(name)
        time_text = f"{seconds:.1f}秒" if seconds is not None else "-"
        tokens = metrics["token_counts"].get(name)
        token_text = f"{tokens['prompt_tokens']} / {tokens['completion_tokens']}" if tokens else "-"
        lines.append(f"    {name:<20} {time_text:>8}  {status:<4}  {token_text}")
    lines.append(f"  token合计：输入{metrics['token_counts']['total_prompt_tokens']}，"
                 f"输出{metrics['token_counts']['total_completion_tokens']}")
    lines.append(f"  上下文：检索{meta['total_chunks_retrieved']}条 -> 去重后{meta['unique_chunks_after_dedup']}条"
                 f" -> 选中{meta['chunks_selected']}条 -> 实际留在上下文{meta.get('chunks_in_context', '?')}条，"
                 f"约{meta['estimated_tokens']}个token")
    if metrics["warnings"]:
        lines.append("  警告：")
        for w in metrics["warnings"]:
            lines.append(f"    - {w}")

    if show_intermediate:
        inter = result["intermediate_results"]
        lines.append("")
        lines.append("【中间结果】")
        evaluation = inter["evidence_evaluation"]
        if evaluation is None:
            lines.append("  证据评估：（未进行或失败）")
        else:
            lines.append(f"  证据评估（解析出的JSON）：{evaluation['parsed']}")
        if inter["context_filter"] is not None:
            info = inter["context_filter"]
            lines.append(f"  上下文筛选：共{info['total_documents']}个文档，留下编号{info['kept_ids']}，"
                         f"去掉{[(r['id'], r['title'][:40]) for r in info['removed']]}"
                         + ("（评估器判全部不相关，已保留全部）" if info["fallback_all_irrelevant"] else ""))
        lines.append(f"  答案草稿：{inter['draft_answer']}")
        lines.append(f"  审查意见：{inter['review_feedback']}")
        if inter["citation_check"] is not None:
            lines.append(f"  引用检查：{inter['citation_check']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 直接运行：用3条内置的示例文档快速试一遍完整流程（不需要加载检索模型，只需要Ollama）
# 注意：下面的文本是为了演示流程自己写的示例，不是真实文献原文
# ---------------------------------------------------------------------------
# journal和pub_year放在文档字典最外层，跟12_reranker.py输出的格式一样，14_context_assembler.py会把它们补进metadata
DEMO_DOCS = [
    {"chunk_id": "demo_1", "final_score": 0.92,
     "text": "Metformin use in patients with type 2 diabetes was associated with a lower risk of cardiovascular events "
             "in several cohort studies. The benefit may be partly independent of glycemic control. "
             "Randomized evidence remains limited.",
     "metadata": {"source_title": "Example paper 1: Metformin and cardiovascular outcomes in type 2 diabetes",
                  "section_title": "Results", "doc_id": "demo_doc_1"},
     "journal": "Example Journal of Cardiology", "pub_year": 2021},
    {"chunk_id": "demo_2", "final_score": 0.85,
     "text": "AMPK activation by metformin improves insulin sensitivity and may reduce inflammation in vascular tissue. "
             "These mechanisms were mainly studied in cell and animal models.",
     "metadata": {"source_title": "Example paper 2: Mechanisms of metformin action",
                  "section_title": "Discussion", "doc_id": "demo_doc_2"},
     "journal": "Example Metabolism Reviews", "pub_year": 2019},
    {"chunk_id": "demo_3", "final_score": 0.60,
     "text": "Zebrafish hair cell regeneration depends on supporting cell proliferation after damage. "
             "This study did not involve any drug treatment.",
     "metadata": {"source_title": "Example paper 3: Hair cell regeneration in zebrafish",
                  "section_title": "Introduction", "doc_id": "demo_doc_3"},
     "journal": "Example Developmental Biology", "pub_year": 2018},
]

if __name__ == "__main__":
    pipeline = MedicalGenerationPipeline()
    print(pipeline.llm.connection_message)
    if not pipeline.llm.available:
        raise SystemExit(1)

    demo_result = pipeline.generate("What effects does metformin have on cardiovascular diseases?", DEMO_DOCS)
    print(format_result_report(demo_result))
