"""
生成模块与提示词工程 第一部分 - 上下文组装器

前面两周的检索+重排序模块（11_multi_path_retriever.py、12_reranker.py）产出的是
一份参考文档列表，但这份列表还不能直接喂给生成模型：里面可能有内容重复的片段、
可能全挤在一两篇文献里、也没有限制总长度。这个模块负责把检索结果转换、去重、
挑选、拼接成一段适合放进prompt里的上下文文本，同时记录这个过程的统计信息（元数据），
方便以后排查"生成的答案为什么没引用某篇文献"这类问题。

在整个流程里是第14步（生成部分的开头）。这个文件只定义ContextAssembler类，自己不能直接运行，
由16_verify_generation_prep.py、17_demo_context_assembler.py、20_medical_generation_pipeline.py
等脚本用importlib导入使用。
输入：检索+重排序产出的字典列表（用到text、metadata、chunk_id，以及final_score/fused_score/vector_score
      其中之一当相关性分数，期刊和年份在字典最外层的journal、pub_year）
输出：assemble()返回的字典：context_text（拼好的上下文文本）、metadata（统计信息）、
      selected_chunks、included_chunks
依赖：re、dataclasses是标准库；没传tokenizer时才会用到transformers
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DocumentChunk:
    """组装上下文时用的统一chunk格式：正文、metadata、相关性分数、来源（文献标题）、chunk_id"""
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    chunk_id: str


# Jaccard相似度超过这个值就认为两个chunk内容基本重复了，去重的时候只留相关性更高的那个
JACCARD_DEDUP_THRESHOLD = 0.7

# 同一个来源最多选几个chunk进最终上下文，超过的话优先级往后降，跟前两周检索那边做
# 多样性控制（MAX_PER_SOURCE）是一个思路
MAX_CHUNKS_PER_SOURCE = 2

# 构建好的上下文超长需要截断时，先只在最后这百分之多少的文本范围里找句号，
# 这样能保证是在一个完整句子结束的地方截断，不会把句子从中间切断
TRUNCATE_SEARCH_WINDOW_RATIO = 0.1

# 最后10%里一个句号都没有的话（一句话比较长的时候很常见），放宽到最后这么多再找一次，
# 还是找不到才放弃。这是在任务要求（后10%）基础上多加的一个兜底，不然实测发现
# 经常会停在句子中间
TRUNCATE_FALLBACK_WINDOW_RATIO = 0.5

# tokenizer没有传进来的时候，默认复用12_reranker.py里已经下载过的bge-reranker-base的
# tokenizer来估算token数。注意这不是真正要接的生成模型(比如qwen3:8b)自己的tokenizer，
# 只是手头已经有现成的、不用再多下载一个模型，估出来的token数跟生成模型实际的分词
# 结果会有误差，等真正接上生成模型之后应该换成那个模型自己的tokenizer
DEFAULT_TOKENIZER_MODEL_NAME = "BAAI/bge-reranker-base"


class ContextAssembler:
    """把检索结果去重、挑选、拼接并截断成适合放进prompt的上下文，主入口是assemble()"""

    def __init__(self, tokenizer=None, max_context_tokens=2000):
        # 加载tokenizer：如果外面已经有现成的tokenizer（比如reranker.tokenizer）就直接用，
        # 不传的话自己加载一份
        if tokenizer is None:
            from transformers import AutoTokenizer
            print(f"加载tokenizer用于估算token数：{DEFAULT_TOKENIZER_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL_NAME)
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def estimate_tokens(self, text):
        # 估算文本token数量
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _to_document_chunk(self, doc):
        # 转换文档格式：把检索+重排序那边产出的字典格式转成DocumentChunk。
        # 相关性优先用重排序后的final_score，没有的话退回fused_score，
        # 再没有就退回vector_score，这样不管是接reranker之后的结果还是
        # 只接了MultiPathRetriever的结果都能用
        relevance_score = doc.get("final_score")
        if relevance_score is None:
            relevance_score = doc.get("fused_score")
        if relevance_score is None:
            relevance_score = doc.get("vector_score", 0.0)

        # 复制一份metadata再用，避免改到原来的字典。期刊和年份是重排序那一步加在
        # 文档字典最外层的（不在metadata里），这里补进来，后面列参考文献的时候要用
        metadata = dict(doc.get("metadata", {}) or {})
        for key in ("journal", "pub_year"):
            if doc.get(key) is not None:
                metadata.setdefault(key, doc[key])
        source = metadata.get("source_title", "unknown")

        return DocumentChunk(
            text=doc.get("text", ""),
            metadata=metadata,
            relevance_score=relevance_score,
            source=source,
            chunk_id=doc.get("chunk_id", ""),
        )

    def _jaccard_similarity(self, text_a, text_b):
        # 计算两段文本的Jaccard相似度：按词切开变成两个集合，交集大小除以并集大小
        words_a = set(re.findall(r"[a-z0-9]+", text_a.lower()))
        words_b = set(re.findall(r"[a-z0-9]+", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    def _deduplicate(self, chunks):
        # 计算文本相似性并去重：按相关性从高到低排一遍，依次跟已经留下的chunk两两
        # 算Jaccard相似度，太像的就跳过。因为是按相关性从高到低走的，先留下的
        # 本来就是相关性更高的那个，不需要额外再比一次谁留谁
        chunks_sorted = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        unique_chunks = []
        for chunk in chunks_sorted:
            is_duplicate = False
            for kept in unique_chunks:
                similarity = self._jaccard_similarity(chunk.text, kept.text)
                if similarity >= JACCARD_DEDUP_THRESHOLD:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(chunk)
        return unique_chunks

    def _select_diverse_chunks(self, chunks):
        # 优先选择高相关性文档，并考虑多样化：按相关性排序，正常往里选，
        # 但如果某个来源已经选够了MAX_CHUNKS_PER_SOURCE个，这个来源后面的chunk
        # 就先放进"降级"的队列，等正常优先级的选完了再按相关性顺序补上
        chunks_sorted = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        normal_priority = []
        lowered_priority = []
        source_count = {}

        for chunk in chunks_sorted:
            count = source_count.get(chunk.source, 0)
            if count < MAX_CHUNKS_PER_SOURCE:
                normal_priority.append(chunk)
                source_count[chunk.source] = count + 1
            else:
                lowered_priority.append(chunk)

        return normal_priority + lowered_priority

    def _cut_to_token_limit(self, text, max_tokens):
        # 二分查找：找出最长的、token数不超过预算的前缀。
        # （最开始是先按比例粗截一刀、再每次往回收200个字符，实测发现步子太大，
        # 预算120 token的时候最后只剩下68 token，白白浪费了将近一半的预算）
        low = 0
        high = len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.estimate_tokens(text[:mid]) <= max_tokens:
                low = mid
            else:
                high = mid - 1
        return text[:low]

    def _find_sentence_end(self, text, start):
        # 从后往前，在text[start:]里找最后一个句子结束的位置，找不到返回-1。
        # 中文的。！？直接算；英文的 . ! ? 要后面跟着空白（或者正好是结尾）才算，
        # 这样"2.5"、"HMG-CoA"这种中间带点的就不会被误当成句号
        for i in range(len(text) - 1, start - 1, -1):
            ch = text[i]
            if ch in "。！？":
                return i
            if ch in ".!?" and (i == len(text) - 1 or text[i + 1].isspace()):
                return i
        return -1

    def _trim_to_sentence_end(self, text):
        # 在前缀的后10%范围里找句号，从那里截断；找不到的话放宽到后50%再找一次；
        # 都找不到就保持原样（宁可停在句子中间，也不能把保留的内容砍掉一大半）。
        # 返回(截断后的文本, 实际用到的搜索范围比例)，没找到句号时比例是None
        for ratio in (TRUNCATE_SEARCH_WINDOW_RATIO, TRUNCATE_FALLBACK_WINDOW_RATIO):
            start = int(len(text) * (1 - ratio))
            end_pos = self._find_sentence_end(text, start)
            if end_pos != -1:
                return text[: end_pos + 1].strip(), ratio
        return text.strip(), None

    def _truncate_at_sentence_boundary(self, text, max_tokens):
        # 构建上下文字符串并限制长度：没超长直接返回；超长的话先精确截出
        # 预算以内最长的前缀，再在前缀结尾部分找句号，保证在完整句子处结束
        if self.estimate_tokens(text) <= max_tokens:
            return text

        prefix = self._cut_to_token_limit(text, max_tokens)
        trimmed, _ = self._trim_to_sentence_end(prefix)
        return trimmed

    def _analyze_sources(self, chunks):
        # 统计选中的chunk各来自哪些文献、各有几条，方便看最后上下文的来源分布
        sources = {}
        for chunk in chunks:
            sources[chunk.source] = sources.get(chunk.source, 0) + 1
        return sources

    def assemble(self, retrieved_docs, max_context_tokens=None):
        """
        组装上下文。

        retrieved_docs: 检索+重排序模块（11_multi_path_retriever.py / 12_reranker.py）
                         产出的参考文档字典列表
        max_context_tokens: 上下文token预算，不传就用初始化时的max_context_tokens

        返回字典：context_text、metadata（各步统计）、selected_chunks（截断前挑出的）、
        included_chunks（截断后真正留在上下文里的）
        """
        max_tokens = max_context_tokens if max_context_tokens is not None else self.max_context_tokens

        # 转换文档格式
        chunks = [self._to_document_chunk(doc) for doc in retrieved_docs]

        # 计算文本相似性并去重
        unique_chunks = self._deduplicate(chunks)

        # 按照相关性排序，优先选择高相关性文档，并考虑多样化
        selected_chunks = self._select_diverse_chunks(unique_chunks)

        # 构建上下文字符串并限制长度
        # 每个chunk前面加一行"[来源: 文献标题]"，后面提示词要求模型按这个来源标注引用，
        # 而且_find_included_chunks也是靠这个标题行的格式来判断chunk有没有被截掉的，改的话两处要一起改
        pieces = [f"[来源: {chunk.source}]\n{chunk.text}" for chunk in selected_chunks]
        full_context = "\n\n".join(pieces)
        final_context = self._truncate_at_sentence_boundary(full_context, max_tokens)

        # 截断之后，排在后面的chunk可能整个被截掉了。selected_chunks是截断之前挑出来的，
        # included_chunks才是真正留在上下文文本里的那些（顺序和上下文里的一致）
        included_chunks = self._find_included_chunks(selected_chunks, final_context)

        # 补充元数据
        context_metadata = {
            "total_chunks_retrieved": len(retrieved_docs),
            "unique_chunks_after_dedup": len(unique_chunks),
            "chunks_selected": len(selected_chunks),
            "chunks_in_context": len(included_chunks),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected_chunks),
        }

        return {
            "context_text": final_context,
            "metadata": context_metadata,
            "selected_chunks": selected_chunks,
            "included_chunks": included_chunks,
        }

    def _find_included_chunks(self, selected_chunks, final_context):
        # 判断每个chunk有没有留在最终上下文里：上下文是各个chunk的"标题行+正文"按顺序用空行
        # 拼起来再从后面截断的，所以只要算出每个chunk在拼接文本里的起始位置，
        # 起始位置加上标题行之后还有字符留着，就说明这个chunk至少保留了一部分正文
        included = []
        position = 0
        for chunk in selected_chunks:
            header = f"[来源: {chunk.source}]\n"
            if position + len(header) < len(final_context):
                included.append(chunk)
            position += len(header) + len(chunk.text) + len("\n\n")
        return included
