"""
离线验证脚本：检查 LLMGenerator（19_llm_generator.py）和 MedicalGenerationPipeline
（20_medical_generation_pipeline.py）的逻辑对不对

不需要启动Ollama，也不需要加载任何模型。做法是写一个"假的Ollama"：继承LLMGenerator，
只把真正发HTTP请求的那个函数（_request）换成按事先写好的剧本返回回复，
其他代码（拼提示词、去掉<think>、提取和修补JSON、六个阶段的流程、后处理）都是真的在跑。
这样可以故意构造各种情况——JSON被截断、评估器全判不相关、审查失败、模型引用了不存在的文献等——
看流水线是不是都能正确处理。

用法：python 22_verify_generation_logic.py
没有命令行参数。组装器用的是下面写的FakeTokenizer，所以不会去加载bge-reranker-base。
输出：每项检查打印 [PASS]/[FAIL]，最后汇总PASS和FAIL的个数，有失败的会把名字列出来；不写文件。
（测试2和测试7里各有一次故意去连 127.0.0.1:1 这种连不上的地址，用来测"连不上Ollama"的情况，不会真的访问网络）

注意：这个脚本只能验证"流程和处理逻辑"，验证不了真实模型的回答质量。
真实模型的效果要用 21_test_generation_pipeline.py 来看。
"""

import importlib
import json

_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
_m_llm_generator = importlib.import_module("19_llm_generator")
LLMGenerator = _m_llm_generator.LLMGenerator
_m_medical_generation_pipeline = importlib.import_module("20_medical_generation_pipeline")
MedicalGenerationPipeline = _m_medical_generation_pipeline.MedicalGenerationPipeline
format_result_report = _m_medical_generation_pipeline.format_result_report
DISCLAIMER_KEYWORD = _m_medical_generation_pipeline.DISCLAIMER_KEYWORD
NO_EVIDENCE_ANSWER = _m_medical_generation_pipeline.NO_EVIDENCE_ANSWER
STAGE_NAMES = _m_medical_generation_pipeline.STAGE_NAMES

results = []


def check(name, passed, detail=""):
    """记录并打印一项检查的结果（PASS/FAIL）"""
    status = "PASS" if passed else "FAIL"
    results.append((name, status))
    print(f"  [{status}] {name} {detail}")


class FakeTokenizer:
    # 假tokenizer：每个空格分出来的词算1个token，长词每4个字符多算1个，只用来让组装器能跑起来
    def encode(self, text, add_special_tokens=False):
        tokens = []
        for w in text.split():
            tokens.extend([0] * max(1, (len(w) + 3) // 4))
        return tokens


TITLE_1 = "Metformin and cardiovascular outcomes in type 2 diabetes"
TITLE_2 = "Mechanisms of metformin action in vascular tissue"
TITLE_3 = "Hair cell regeneration in zebrafish"


def make_docs():
    """造3条测试文档：d1、d2讲二甲双胍和心血管，d3讲斑马鱼（故意跟问题无关，用来测筛选）"""
    return [
        {"chunk_id": "d1", "final_score": 0.9, "journal": "J Cardiol", "pub_year": 2021,
         "text": "Metformin was associated with fewer cardiovascular events in cohort studies of type 2 diabetes.",
         "metadata": {"source_title": TITLE_1, "section_title": "Results", "doc_id": "doc1"}},
        {"chunk_id": "d2", "final_score": 0.8, "journal": "Metab Rev", "pub_year": 2019,
         "text": "AMPK activation by metformin improves insulin sensitivity, mostly shown in animal models.",
         "metadata": {"source_title": TITLE_2, "section_title": "Discussion", "doc_id": "doc2"}},
        {"chunk_id": "d3", "final_score": 0.5, "journal": "Dev Biol", "pub_year": 2018,
         "text": "Zebrafish supporting cells proliferate after hair cell damage and no drug was tested.",
         "metadata": {"source_title": TITLE_3, "section_title": "Intro", "doc_id": "doc3"}},
    ]


# 四个阶段的system prompt里各有一句特征话，用它判断当前请求是哪个阶段
STAGE_MARKERS = {
    "evidence_evaluator": "医学文献评估助手",
    "answer_generator": "医学文献问答助手",
    "critical_reviewer": "严格的医学内容审查员",
    "final_assembler": "最终答案整合助手",
}

EVAL_JSON_OK = json.dumps({
    "overall": "部分充分", "reason": "有队列研究但缺少随机试验",
    "documents": [
        {"id": 1, "relevant": True, "strength": "中", "comment": "队列研究支持心血管获益"},
        {"id": 2, "relevant": True, "strength": "弱", "comment": "机制研究，主要来自动物模型"},
        {"id": 3, "relevant": False, "strength": "弱", "comment": "讨论斑马鱼，与问题无关"},
    ],
    "gaps": "缺少随机对照试验证据",
}, ensure_ascii=False)

# 草稿里第二处引用的标题故意多写了一个s（tissues），带"**"加粗，而且结尾带着模型自己写的免责声明，
# 用来测模糊匹配、格式整理和免责声明去重
DRAFT_TEXT = (f"二甲双胍在2型糖尿病患者中与较少的心血管事件相关[来源: {TITLE_1}]。"
              f"机制研究提示它可能改善胰岛素敏感性，但证据主要来自动物模型[来源: Mechanisms of metformin action in vascular tissues]。\n\n"
              f"**注意**：以上证据多为观察性研究。\n"
              f"本回答基于检索到的文献证据整理生成，不构成医疗建议，具体诊疗请遵医嘱。")
REVIEW_TEXT = "审查意见：机制研究的结论被说成了临床获益，需要调低表述；其他没有问题。"
FINAL_TEXT = (f"现有观察性研究提示，二甲双胍与2型糖尿病患者较少的心血管事件相关[来源: {TITLE_1}]。"
              f"机制方面的证据主要来自动物模型，尚不能直接等同于临床获益[来源: {TITLE_2}]。\n"
              f"* 缺少随机对照试验证据\n"
              f"本回答基于检索到的文献证据整理生成，不构成医疗建议，具体诊疗请遵医嘱。")


class ScriptedLLM(LLMGenerator):
    """假的Ollama：按剧本返回每个阶段的回复，并记录每次收到的请求"""

    def __init__(self, script):
        super().__init__(test_connection=False)
        self.available = True
        self.script = script          # {阶段名: 回复文字，或者"ERROR"表示这个阶段请求失败}
        self.calls = []               # [(阶段名, 发出的payload)]

    def _request(self, method, path, payload=None, timeout=None):
        # 靠system prompt里的特征话判断当前是哪个阶段；merge_system_into_user=True时没有system消息，就会判成unknown
        system_text = payload["messages"][0]["content"] if payload["messages"][0]["role"] == "system" else ""
        stage = "unknown"
        for name, marker in STAGE_MARKERS.items():
            if marker in system_text:
                stage = name
        self.calls.append((stage, payload))

        reply = self.script.get(stage, "")
        if reply == "ERROR":
            raise TimeoutError("模拟请求超时")
        return {"message": {"role": "assistant", "content": reply},
                "prompt_eval_count": 100, "eval_count": 50, "done_reason": "stop"}

    def called_stages(self):
        """按调用顺序返回被调用过的阶段名列表"""
        return [name for name, _ in self.calls]

    def user_prompt_of(self, stage):
        """返回某个阶段第一次被调用时发出的user消息内容，这个阶段没被调用就返回None"""
        for name, payload in self.calls:
            if name == stage:
                return payload["messages"][-1]["content"]
        return None


FULL_SCRIPT = {
    "evidence_evaluator": EVAL_JSON_OK,
    "answer_generator": DRAFT_TEXT,
    "critical_reviewer": REVIEW_TEXT,
    "final_assembler": FINAL_TEXT,
}


def make_pipeline(script, **kwargs):
    """用剧本造一个假LLM和用假tokenizer的组装器，组成流水线；kwargs（比如enable_review=False）传给MedicalGenerationPipeline。返回(流水线, 假LLM)"""
    llm = ScriptedLLM(script)
    assembler = ContextAssembler(tokenizer=FakeTokenizer(), max_context_tokens=2000)
    return MedicalGenerationPipeline(assembler=assembler, llm=llm, **kwargs), llm


def test_json_repair():
    """测试1：extract_json/repair_json能不能处理各种坏掉的JSON（多余文字、代码块、多余逗号、被截断、中文逗号、没有JSON）"""
    print("\n--- 测试1：JSON提取和修补（LLMGenerator.extract_json）---")
    llm = LLMGenerator(test_connection=False)
    cases = [
        ("正常JSON", '{"a": 1, "b": [1, 2]}', {"a": 1, "b": [1, 2]}),
        ("前后有多余文字", '结果如下：{"a": 1} 以上', {"a": 1}),
        ("markdown代码块", '```json\n{"a": 1}\n```', {"a": 1}),
        ("多余的逗号", '{"a": 1, "b": [1, 2,],}', {"a": 1, "b": [1, 2]}),
        ("少了结尾的括号", '{"a": [1, 2', {"a": [1, 2]}),
        ("字符串写到一半被截断", '{"a": "没写完', {"a": "没写完"}),
        ("键写到一半被截断", '{"a": 1, "doc": [{"id": 1}, {"id": 2, "rel', {"a": 1, "doc": [{"id": 1}, {"id": 2}]}),
        ("冒号后面没有值", '{"a": 1, "b":', {"a": 1, "b": None}),
        ("中文逗号", '{"a": 1，"b": 2}', {"a": 1, "b": 2}),
    ]
    for name, text, expected in cases:
        obj, ok = llm.extract_json(text)
        check(f"JSON：{name}", ok and obj == expected, f"(得到 {obj})")
    obj, ok = llm.extract_json("这里面没有JSON")
    check("JSON：没有JSON时返回失败，不报错", (not ok) and obj is None)


def test_generate_basics():
    """测试2：generate()的基本行为——去掉<think>、参数传给Ollama、token数记录、JSON格式要求、system合并进user、连不上不抛异常、批量生成"""
    print("\n--- 测试2：generate()的基本行为 ---")
    llm = ScriptedLLM({"answer_generator": "<think>先想一想</think>最终答案"})
    r = llm.generate("问题", system_prompt="你是医学文献问答助手", temperature=0.2, max_tokens=77)
    sent = llm.calls[0][1]
    check("<think>思考内容被去掉", r["text"] == "最终答案", f"(得到 {r['text']!r})")
    check("temperature和max_tokens传给了Ollama",
          sent["options"]["temperature"] == 0.2 and sent["options"]["num_predict"] == 77)
    check("token数记录下来了", r["prompt_tokens"] == 100 and r["completion_tokens"] == 50)

    llm.script = {"unknown": "{}"}
    llm.generate("问题", system_prompt="随便", json_format={"x": "说明"})
    system_text = llm.calls[-1][1]["messages"][0]["content"]
    check("要求JSON时，system prompt末尾加上了格式要求", "JSON" in system_text and '"x"' in system_text)

    llm.merge_system_into_user = True
    llm.generate("用户问题", system_prompt="系统提示")
    messages = llm.calls[-1][1]["messages"]
    check("merge_system_into_user=True时，system拼进了用户消息",
          len(messages) == 1 and messages[0]["role"] == "user" and "系统提示" in messages[0]["content"])

    dead = LLMGenerator(base_url="http://127.0.0.1:1", test_connection=True)
    r = dead.generate("问题")
    check("连不上Ollama时返回失败信息而不是抛异常", (not dead.available) and (not r["success"]) and r["error"])

    batch = ScriptedLLM({"unknown": "ok"}).generate_batch(["a", {"prompt": "b", "max_tokens": 5}], verbose=False)
    check("批量生成：结果数量和顺序对", len(batch) == 2 and [x["index"] for x in batch] == [0, 1])


def test_full_flow():
    """测试3：评估和审查都开的完整流程——调用顺序、各阶段拿到的提示词内容、筛选、引用编号、免责声明、参考文献、sources和报告"""
    print("\n--- 测试3：完整流程（评估 + 审查都开）---")
    pipeline, llm = make_pipeline(FULL_SCRIPT)
    result = pipeline.generate("What effects does metformin have on cardiovascular diseases?", make_docs())

    required = ["query", "answer", "context_metadata", "generation_metrics", "intermediate_results", "sources", "timestamp"]
    check("结果字典包含约定的所有字段", all(k in result for k in required))
    metrics = result["generation_metrics"]
    check("generation_metrics包含约定的字段",
          all(k in metrics for k in ["total_time_seconds", "stage_times", "token_counts", "stage_success"]))
    check("四次模型调用，顺序是 评估->草稿->审查->最终",
          llm.called_stages() == ["evidence_evaluator", "answer_generator", "critical_reviewer", "final_assembler"],
          f"(实际 {llm.called_stages()})")

    check("六个阶段都成功", all(metrics["stage_success"][n] is True for n in STAGE_NAMES),
          f"({metrics['stage_success']})")
    check("token合计 = 四次调用之和", metrics["token_counts"]["total_prompt_tokens"] == 400
          and metrics["token_counts"]["total_completion_tokens"] == 200)

    eval_prompt = llm.user_prompt_of("evidence_evaluator")
    check("评估提示词里的文档带编号【文档1】【文档3】", "【文档1】" in eval_prompt and "【文档3】" in eval_prompt)

    gen_prompt = llm.user_prompt_of("answer_generator")
    check("不相关的文档（斑马鱼）被筛出了答案生成的上下文", "Zebrafish" not in gen_prompt and "zebrafish" not in gen_prompt)
    check("相关的文档留在了上下文里", "AMPK activation" in gen_prompt and "cohort studies" in gen_prompt)
    check("评估结果（总体判断、证据缺口）传给了答案生成器",
          "部分充分" in gen_prompt and "缺少随机对照试验证据" in gen_prompt)
    check("传给答案生成器的评估文字里没有被筛掉的文档", TITLE_3 not in gen_prompt)

    info = result["intermediate_results"]["context_filter"]
    check("筛选记录：留下[1,2]，去掉第3个", info["kept_ids"] == [1, 2] and [r["id"] for r in info["removed"]] == [3])

    review_prompt = llm.user_prompt_of("critical_reviewer")
    check("审查器收到的是草稿和筛选后的上下文", "二甲双胍在2型糖尿病" in review_prompt and "Zebrafish" not in review_prompt)
    final_prompt = llm.user_prompt_of("final_assembler")
    check("最终组装器收到了草稿和审查意见", "二甲双胍在2型糖尿病" in final_prompt and "调低表述" in final_prompt)

    answer = result["answer"]
    check("答案里的[来源: 标题]全部换成了编号", "[来源" not in answer and "[1]" in answer and "[2]" in answer, f"\n{answer}")
    check("免责声明恰好出现一次", answer.count(DISCLAIMER_KEYWORD) == 1)
    check("免责声明在答案末尾", answer.rstrip().split("\n")[-1].startswith("免责声明"))
    check("列表符号统一成了 - ", "\n- 缺少随机对照试验证据" in answer and "* 缺少" not in answer)
    check("附上了参考文献，且包含期刊和年份", "参考文献" in answer and "J Cardiol" in answer and "2019" in answer)
    check("参考文献里没有被筛掉的文献", TITLE_3 not in answer)

    sources = result["sources"]
    check("sources：2条，都标记为已引用", len(sources) == 2 and all(s["cited"] for s in sources))
    check("sources：编号和标题对得上", sources[0]["title"] == TITLE_1 and sources[0]["ref_id"] == 1)
    check("答案长度已记录", metrics["answer_length_chars"] == len(answer))

    report = format_result_report(result)
    check("format_result_report能生成报告，包含答案和统计", "最终答案" in report and "生成统计" in report
          and "answer_generation" in report)


def test_fuzzy_citation():
    """测试4：模型引用的标题和原标题略有出入时能模糊匹配上；完全对不上的引用保持原样并给警告"""
    print("\n--- 测试4：模型引用的标题和原标题略有出入（模糊匹配）---")
    pipeline, llm = make_pipeline(FULL_SCRIPT, enable_review=False)
    result = pipeline.generate("q", make_docs())
    # 草稿里引用的第二篇标题故意多写了一个s（tissues），应该仍然匹配上编号2
    check("多一个字母的标题也能匹配到[2]", "[2]" in result["answer"] and "[来源" not in result["answer"])

    script = dict(FULL_SCRIPT)
    script["answer_generator"] = "这是一个结论[来源: A completely different paper about something else]。"
    pipeline, llm = make_pipeline(script, enable_review=False)
    result = pipeline.generate("q", make_docs())
    check("对不上任何来源的引用：保持原样并给出警告",
          "[来源: A completely different paper" in result["answer"]
          and len(result["intermediate_results"]["citation_check"]["unmatched_citations"]) == 1
          and any("对不上" in w for w in result["generation_metrics"]["warnings"]))


def test_evaluation_variants():
    """测试5：证据评估的几种情况——JSON被截断、没输出JSON、全判不相关、id/relevant写成字符串、关闭评估"""
    print("\n--- 测试5：证据评估的各种情况 ---")
    # 评估器的JSON被max_tokens截断了（这里把"gaps"及后面的部分砍掉，再去掉末尾的逗号，模拟写到一半就停）
    script = dict(FULL_SCRIPT)
    script["evidence_evaluator"] = EVAL_JSON_OK[: EVAL_JSON_OK.index('"gaps"')] .rstrip().rstrip(",")
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    check("评估JSON被截断：补全后仍然能筛选", result["intermediate_results"]["context_filter"] is not None
          and result["intermediate_results"]["context_filter"]["kept_ids"] == [1, 2])
    check("评估JSON被截断：评估阶段算成功", result["generation_metrics"]["stage_success"]["evidence_evaluation"] is True)

    # 评估器没有输出JSON
    script = dict(FULL_SCRIPT)
    script["evidence_evaluator"] = "这些证据大体够用，第3个文档不相关。"
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    gen_prompt = llm.user_prompt_of("answer_generator")
    check("评估器没输出JSON：评估阶段标记失败", result["generation_metrics"]["stage_success"]["evidence_evaluation"] is False)
    check("评估器没输出JSON：不筛选，保留全部文档", "Zebrafish" in gen_prompt)
    check("评估器没输出JSON：流程继续，原话传给答案生成器", "这些证据大体够用" in gen_prompt
          and result["generation_metrics"]["stage_success"]["answer_generation"] is True)

    # 评估器说所有文档都不相关
    script = dict(FULL_SCRIPT)
    script["evidence_evaluator"] = json.dumps({"overall": "不充分", "documents": [
        {"id": 1, "relevant": False}, {"id": 2, "relevant": "否"}, {"id": 3, "relevant": False}]})
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    gen_prompt = llm.user_prompt_of("answer_generator")
    check("评估器全判不相关：保留全部文档（避免上下文为空）并给出警告",
          "cohort studies" in gen_prompt and result["intermediate_results"]["context_filter"]["fallback_all_irrelevant"]
          and any("全部" in w for w in result["generation_metrics"]["warnings"]))

    # relevant写成了字符串，id写成了"文档3"
    script = dict(FULL_SCRIPT)
    script["evidence_evaluator"] = json.dumps({"overall": "部分充分", "documents": [
        {"id": "文档1", "relevant": "是"}, {"id": "文档3", "relevant": "false"}]}, ensure_ascii=False)
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    check("relevant和id写成字符串也能识别", "Zebrafish" not in llm.user_prompt_of("answer_generator"))

    # 关闭评估
    pipeline, llm = make_pipeline(FULL_SCRIPT, enable_evaluation=False)
    result = pipeline.generate("q", make_docs())
    check("关闭评估：不调用评估器，评估阶段显示跳过",
          "evidence_evaluator" not in llm.called_stages() and result["generation_metrics"]["stage_success"]["evidence_evaluation"] is None)
    check("关闭评估：所有文档都留在上下文里", "Zebrafish" in llm.user_prompt_of("answer_generator"))


def test_review_variants():
    """测试6：审查和最终组装的几种情况——关闭审查、审查请求失败、最终组装失败，都应该退回草稿而不是中断"""
    print("\n--- 测试6：审查和最终答案的各种情况 ---")
    pipeline, llm = make_pipeline(FULL_SCRIPT, enable_review=False)
    result = pipeline.generate("q", make_docs())
    check("关闭审查：不调用审查器和最终组装器",
          llm.called_stages() == ["evidence_evaluator", "answer_generator"])
    check("关闭审查：直接使用草稿（草稿的内容出现在答案里）", "较少的心血管事件相关" in result["answer"]
          and "调低" not in result["answer"])
    stages = result["generation_metrics"]["stage_success"]
    check("关闭审查：审查和最终组装阶段显示跳过", stages["critical_review"] is None and stages["final_assembly"] is None)

    script = dict(FULL_SCRIPT)
    script["critical_reviewer"] = "ERROR"
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    stages = result["generation_metrics"]["stage_success"]
    check("审查失败：回退到草稿，流程不中断", stages["critical_review"] is False and "final_assembler" not in llm.called_stages()
          and "较少的心血管事件相关" in result["answer"] and "免责声明" in result["answer"])

    script = dict(FULL_SCRIPT)
    script["final_assembler"] = "ERROR"
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    stages = result["generation_metrics"]["stage_success"]
    check("最终组装失败：回退到草稿", stages["final_assembly"] is False
          and "较少的心血管事件相关" in result["answer"] and stages["post_processing"] is True)


def test_failure_and_edge_cases():
    """测试7：失败和边界情况——草稿失败、没有检索结果、预算很小、答案没有引用、Ollama连不上"""
    print("\n--- 测试7：失败和边界情况 ---")
    script = dict(FULL_SCRIPT)
    script["answer_generator"] = "ERROR"
    pipeline, llm = make_pipeline(script)
    result = pipeline.generate("q", make_docs())
    check("草稿失败：返回失败说明，不再调用审查", result["answer"].startswith("答案生成失败")
          and "critical_reviewer" not in llm.called_stages()
          and result["generation_metrics"]["stage_success"]["answer_generation"] is False)
    check("草稿失败：结果字典结构仍然完整", all(k in result for k in ["sources", "intermediate_results", "generation_metrics"]))

    pipeline, llm = make_pipeline(FULL_SCRIPT)
    result = pipeline.generate("q", [])
    check("没有检索结果：不调用模型，说明没有证据", llm.calls == [] and result["answer"] == NO_EVIDENCE_ANSWER
          and result["generation_metrics"]["stage_success"]["context_assembly"] is False)

    # 预算很小：只有第一个chunk能留在上下文里
    pipeline, llm = make_pipeline(FULL_SCRIPT, max_context_tokens=22)
    result = pipeline.generate("q", make_docs())
    meta = result["context_metadata"]
    check("预算很小：被截掉的chunk不算在上下文里（chunks_in_context < chunks_selected）",
          meta["chunks_in_context"] < meta["chunks_selected"], f"({meta['chunks_in_context']} < {meta['chunks_selected']})")
    check("预算很小：参考来源只列留在上下文里的文献", len(result["sources"]) <= meta["chunks_in_context"])
    eval_prompt = llm.user_prompt_of("evidence_evaluator")
    check("预算很小：评估提示词里只有留下的文档编号", "【文档1】" in eval_prompt and "【文档3】" not in eval_prompt)

    # 模型的回答里没有任何引用
    script = dict(FULL_SCRIPT)
    script["answer_generator"] = "没有引用的回答。"
    pipeline, llm = make_pipeline(script, enable_review=False)
    result = pipeline.generate("q", make_docs())
    check("答案没有引用：参考文献部分改成'没有标注具体引用'的说明", "没有标注具体引用" in result["answer"]
          and not any(s["cited"] for s in result["sources"]) and result["intermediate_results"]["citation_check"]["has_citation"] is False)

    # 模型连不上
    dead_llm = LLMGenerator(base_url="http://127.0.0.1:1")
    assembler = ContextAssembler(tokenizer=FakeTokenizer())
    pipeline = MedicalGenerationPipeline(assembler=assembler, llm=dead_llm)
    result = pipeline.generate("q", make_docs())
    check("Ollama连不上：返回失败说明，不抛异常", result["answer"].startswith("答案生成失败"))


if __name__ == "__main__":
    test_json_repair()
    test_generate_basics()
    test_full_flow()
    test_fuzzy_citation()
    test_evaluation_variants()
    test_review_variants()
    test_failure_and_edge_cases()

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s in results if s == "PASS")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    print(f"汇总：PASS={n_pass}  FAIL={n_fail}")
    if n_fail:
        print("失败的检查：")
        for name, status in results:
            if status == "FAIL":
                print("  -", name)
