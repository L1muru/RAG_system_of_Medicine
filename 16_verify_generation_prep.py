"""
验证脚本：检查上下文组装器（14_context_assembler.py）和医学提示词模板（15_prompt_templates.py）

这两个模块本身是"库"，直接运行没有任何输出，所以单独写一个验证脚本。分两部分：

第一部分（离线测试）：用自己构造的数据测试各种边界情况（重复内容、同一来源占满、
        预算很小、空输入等），不依赖检索流水线，结果是确定的，几秒钟就能跑完。
第二部分（真实数据测试）：接上前两周的真实检索流水线（查询理解 -> 多路检索 -> 重排序），
        把真实检索结果交给上下文组装器，检查组装结果，再把真实上下文填进四个提示词模板。

用法：
    python 16_verify_generation_prep.py --offline    只跑第一部分
    python 16_verify_generation_prep.py              两部分都跑（第二部分要加载模型和建BM25索引，比较慢）
（命令行只认 --offline 这一个参数，不用argparse，是直接看sys.argv里有没有它）

输出：每项检查打印 [PASS]/[FAIL]/[WARN]，最后汇总三种结果各多少个，并列出失败和警告的检查。
第一部分需要14_context_assembler.py、15_prompt_templates.py，会加载bge-reranker-base的tokenizer；
第二部分还要用10_query_understanding.py、11_multi_path_retriever.py、12_reranker.py和已经建好的向量索引。

注意：这个脚本没有调用生成模型（还没接qwen3:8b），所以只能验证到"喂给模型的输入对不对"，
模型生成的答案质量要等接上模型之后再验证。
"""

import importlib
import sys

_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
DocumentChunk = _m_context_assembler.DocumentChunk
JACCARD_DEDUP_THRESHOLD = _m_context_assembler.JACCARD_DEDUP_THRESHOLD
MAX_CHUNKS_PER_SOURCE = _m_context_assembler.MAX_CHUNKS_PER_SOURCE
_m_prompt_templates = importlib.import_module("15_prompt_templates")
PROMPT_STAGES = _m_prompt_templates.PROMPT_STAGES

# 第二部分用的参数，跟13_retrieval_pipeline.py保持一致
BM25_MAX_CHUNKS = 20000
TEST_QUERIES = [
    "What effects does metformin have on cardiovascular diseases?",
    "MI patients with HTN, what's the aspirin dosage?",
    "What is the relationship between LDL cholesterol and stroke risk?",
]

# 上下文预算（token），跟ContextAssembler的默认值一致
MAX_CONTEXT_TOKENS = 2000

# 假设生成模型的上下文窗口大小。Ollama默认的上下文窗口往往只有几千token（具体取决于
# 版本和显存，需要自己确认，必要时在调用时用num_ctx参数调大），这里先按4096估一个，
# 后面接上模型之后换成真实的数字
ASSUMED_CONTEXT_WINDOW = 4096

# 每个阶段的输入里会塞进上游阶段的输出，估算最坏情况的长度时要把上游阶段的max_tokens加上
UPSTREAM_STAGES = {
    "evidence_evaluator": [],
    "answer_generator": ["evidence_evaluator"],
    "critical_reviewer": ["answer_generator"],
    "final_assembler": ["answer_generator", "critical_reviewer"],
}

results = []


def check(name, passed, detail=""):
    """记录并打印一项检查的结果，passed为True打印PASS，否则FAIL"""
    status = "PASS" if passed else "FAIL"
    results.append((name, status))
    print(f"  [{status}] {name} {detail}")


def warn(name, detail=""):
    """记录并打印一条警告，不算失败，但需要人工看一下"""
    results.append((name, "WARN"))
    print(f"  [WARN] {name} {detail}")


def make_doc(chunk_id, text, source, score):
    """造一条候选文档的字典（chunk编号、文本、来源标题、相关性分数），给离线测试用"""
    # 构造一条跟12_reranker.py输出格式一样的候选文档
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"source_title": source, "doc_id": source},
        "final_score": score,
    }


def check_assembly(label, assembler, result, max_tokens):
    """对一次assemble()的结果做通用检查：token不超预算、元数据自洽、没有重复内容、排序、多样化、截断位置"""
    # 对一次组装的结果做通用检查
    print(f"\n--- 检查组装结果：{label} ---")
    meta = result["metadata"]
    chunks = result["selected_chunks"]
    context = result["context_text"]

    # 1. token预算
    real_tokens = assembler.estimate_tokens(context)
    check("token数不超过预算", real_tokens <= max_tokens, f"({real_tokens} <= {max_tokens})")

    # 2. 元数据自洽
    check("元数据数量关系 total >= unique >= selected",
          meta["total_chunks_retrieved"] >= meta["unique_chunks_after_dedup"] >= meta["chunks_selected"],
          f"({meta['total_chunks_retrieved']} / {meta['unique_chunks_after_dedup']} / {meta['chunks_selected']})")
    check("chunks_selected跟selected_chunks长度一致", meta["chunks_selected"] == len(chunks))
    check("chunk_sources各来源计数之和等于选中数", sum(meta["chunk_sources"].values()) == len(chunks))
    check("estimated_tokens跟重新计算的一致", meta["estimated_tokens"] == real_tokens)

    # 3. 选中的chunk之间不应该有高度重复的内容
    max_sim = 0.0
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            sim = assembler._jaccard_similarity(chunks[i].text, chunks[j].text)
            max_sim = max(max_sim, sim)
    check("选中的chunk两两之间没有重复内容", max_sim < JACCARD_DEDUP_THRESHOLD,
          f"(最大Jaccard相似度={max_sim:.2f}，阈值={JACCARD_DEDUP_THRESHOLD})")

    # 4. 排序：第一条应该是相关性最高的
    if chunks:
        top_score = max(c.relevance_score for c in chunks)
        check("第一条是相关性最高的", chunks[0].relevance_score == top_score)

    # 5. 多样化：同一来源超过上限的那些chunk，必须全部排在没超限的chunk后面
    seen = {}
    lowered_started = False
    order_ok = True
    for c in chunks:
        seen[c.source] = seen.get(c.source, 0) + 1
        if seen[c.source] > MAX_CHUNKS_PER_SOURCE:
            lowered_started = True
        elif lowered_started:
            order_ok = False
    check("同一来源超过上限的chunk都排在后面", order_ok)

    # 6. 截断：如果发生了截断，看是不是停在句子结束的地方
    full_context = "\n\n".join(f"[来源: {c.source}]\n{c.text}" for c in chunks)
    if full_context != context:
        stripped = context.rstrip()
        if stripped and stripped[-1] in ".。!?！？":
            check("发生了截断，且停在句子结束处", True)
        else:
            warn("发生了截断，但没有停在句号处（已知的边界情况：截断点附近后50%范围里都没有句号）",
                 f"结尾是: ...{stripped[-30:]!r}")
    else:
        print("  [INFO] 没有发生截断（内容没超过预算）")


def check_prompts(assembler, question, context_text):
    """检查四个提示词模板：占位符能不能填上，最坏情况的token总开销是否在假设的窗口内。返回填好的user prompt字典"""
    # 把上下文填进四个提示词模板，检查能不能填上、估算每个阶段的token开销
    print("\n--- 检查提示词模板 ---")
    check("四个阶段都在", set(PROMPT_STAGES.keys()) ==
          {"evidence_evaluator", "answer_generator", "critical_reviewer", "final_assembler"})

    values = {
        "question": question,
        "context": context_text,
        "evidence_evaluation": "(此处是证据评估器的输出)",
        "draft_answer": "(此处是答案生成器的输出)",
        "review_feedback": "(此处是批判性审查器的输出)",
    }

    filled = {}
    for key, stage in PROMPT_STAGES.items():
        try:
            user_prompt = stage.user_prompt_template.format(**values)
        except KeyError as e:
            check(f"{key}：模板占位符都能填上", False, f"(模板里的占位符{e}没有对应的值)")
            continue
        filled[key] = user_prompt
        check(f"{key}：模板占位符都能填上，且问题文本出现在提示词里", question in user_prompt)

    # 估算最坏情况下每个阶段的总token开销：
    # 输入（system + 填好的user prompt + 上游阶段的最长输出）+ 本阶段最长输出，要在上下文窗口以内
    for key, stage in PROMPT_STAGES.items():
        if key not in filled:
            continue
        input_tokens = assembler.estimate_tokens(stage.system_prompt) + assembler.estimate_tokens(filled[key])
        upstream_tokens = sum(PROMPT_STAGES[u].max_tokens for u in UPSTREAM_STAGES[key])
        total = input_tokens + upstream_tokens + stage.max_tokens
        detail = (f"(输入{input_tokens} + 上游输出最长{upstream_tokens} + 本阶段输出最长{stage.max_tokens}"
                  f" = {total}，假设的窗口={ASSUMED_CONTEXT_WINDOW})")
        if total <= ASSUMED_CONTEXT_WINDOW:
            check(f"{key}：token总开销在上下文窗口内", True, detail)
        else:
            warn(f"{key}：最坏情况下token总开销超过假设的上下文窗口", detail)

    return filled


def run_offline_tests(assembler):
    """第一部分：用自己构造的数据跑A~E五个用例（重复、同来源占满、预算很小、空输入、字段缺失），最后试填一遍提示词模板"""
    print("=" * 60)
    print("第一部分：离线测试（构造的数据）")
    print("=" * 60)

    # 用例A：内容重复（a3只是比a1多了结尾的空格，应该也被当成重复）
    base = ("Metformin reduces cardiovascular events in patients with type 2 diabetes "
            "by activating AMPK and improving insulin sensitivity.")
    docs = [
        make_doc("a1", base, "Paper S1", 0.9),
        make_doc("a2", base, "Paper S2", 0.8),
        make_doc("a3", base + "  ", "Paper S3", 0.7),
        make_doc("a4", "Warfarin dosing requires regular INR monitoring to balance bleeding and clotting risk.",
                 "Paper S4", 0.5),
    ]
    result = assembler.assemble(docs)
    check_assembly("A 内容重复（3条完全一样 + 1条不同）", assembler, result, MAX_CONTEXT_TOKENS)
    ids = [c.chunk_id for c in result["selected_chunks"]]
    check("A：去重后只剩2条", result["metadata"]["unique_chunks_after_dedup"] == 2,
          f"(实际={result['metadata']['unique_chunks_after_dedup']})")
    check("A：重复的几条里保留的是相关性最高的a1", "a1" in ids and "a2" not in ids and "a3" not in ids, f"(选中={ids})")

    # 用例B：同一来源占满（5条都是Big Paper，只有1条别的来源，而且分数最低，看它能不能被提前）
    topics = [
        "Aspirin inhibits platelet aggregation by irreversibly blocking cyclooxygenase enzymes.",
        "Statins reduce LDL cholesterol through inhibition of HMG-CoA reductase in the liver.",
        "Insulin therapy requires careful glucose monitoring to avoid hypoglycemia in elderly patients.",
        "Warfarin dosing is adjusted according to international normalized ratio measurements.",
        "Lisinopril lowers blood pressure by inhibiting the angiotensin converting enzyme pathway.",
    ]
    docs = [make_doc(f"b{i}", t, "Big Paper", 0.9 - i * 0.1) for i, t in enumerate(topics)]
    docs.append(make_doc("b_other", "Ibuprofen may increase cardiovascular risk when used at high doses for long periods.",
                         "Other Paper", 0.3))
    result = assembler.assemble(docs)
    check_assembly("B 同一来源占满（5条同来源 + 1条其他来源，相关性最低）", assembler, result, MAX_CONTEXT_TOKENS)
    ids = [c.chunk_id for c in result["selected_chunks"]]
    check("B：其他来源的chunk被提到了同一来源超限部分的前面",
          ids.index("b_other") == MAX_CHUNKS_PER_SOURCE, f"(顺序={ids})")

    # 用例C：预算很小
    small_budget = 40
    result = assembler.assemble(docs, max_context_tokens=small_budget)
    check_assembly(f"C 预算很小（{small_budget} token）", assembler, result, small_budget)

    # 用例D：空输入
    result = assembler.assemble([])
    check("D 空输入不报错，上下文为空", result["context_text"] == "" and result["selected_chunks"] == [])
    check("D 空输入的元数据全为0",
          result["metadata"]["total_chunks_retrieved"] == 0 and result["metadata"]["chunks_selected"] == 0)

    # 用例E：字段缺失时的兜底（没有metadata、只有vector_score）
    result = assembler.assemble([{"chunk_id": "e1", "text": "Some text without metadata.", "vector_score": 0.42}])
    chunk = result["selected_chunks"][0]
    check("E 缺少metadata时来源退回unknown", chunk.source == "unknown")
    check("E 缺少final_score/fused_score时相关性退回vector_score", chunk.relevance_score == 0.42)
    check("E 返回的是DocumentChunk类型", isinstance(chunk, DocumentChunk))

    # 提示词模板：用用例B的上下文试填一遍（注意这里的docs还是用例B的那批，因为用例C、D、E没有重新给docs赋值）
    result = assembler.assemble(docs)
    check_prompts(assembler, "What are the cardiovascular effects of common drugs?", result["context_text"])


def run_real_pipeline_tests(assembler, queries):
    """第二部分：对每个查询走真实的检索+重排序，再交给组装器和提示词模板检查，并把结果打印出来供人工查看"""
    print("\n" + "=" * 60)
    print("第二部分：真实数据测试（接前两周的检索流水线）")
    print("=" * 60)

    # 这些库比较重，放在函数里导入，这样--offline的时候不需要它们
    import chromadb
    from sentence_transformers import SentenceTransformer

    _m_query_understanding = importlib.import_module("10_query_understanding")
    process_medical_query = _m_query_understanding.process_medical_query
    CHROMA_DIR = _m_query_understanding.CHROMA_DIR
    COLLECTION_NAME = _m_query_understanding.COLLECTION_NAME
    EMBEDDING_MODEL_NAME = _m_query_understanding.EMBEDDING_MODEL_NAME
    _m_multi_path_retriever = importlib.import_module("11_multi_path_retriever")
    MultiPathRetriever = _m_multi_path_retriever.MultiPathRetriever
    _m_reranker = importlib.import_module("12_reranker")
    Reranker = _m_reranker.Reranker

    print("加载embedding模型、连接向量索引、构建BM25索引、加载重排序模型...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    retriever = MultiPathRetriever(collection, embed_model, max_chunks=BM25_MAX_CHUNKS)
    reranker = Reranker()

    for q in queries:
        print("\n" + "#" * 60)
        print("查询:", q)
        query_info = process_medical_query(q)
        print("识别实体:", query_info["entities"])

        # 多路检索 + 融合，重排序时不截断(top_k=None)，把全部候选交给上下文组装器去去重和挑选
        fused = retriever.retrieve(query_info, top_k_vector=15, top_k_keyword=15, fusion_strategy="rrf", top_k=10)
        reranked = reranker.rerank(query_info, fused, top_k=None)

        result = assembler.assemble(reranked)
        check_assembly(f"真实查询：{q}", assembler, result, MAX_CONTEXT_TOKENS)

        # 额外检查：被选进上下文的chunk里，有没有"实体匹配=False"的（没提到查询里识别出的药物/疾病）
        # 这类内容是上周发现reranker容易误判的，进了上下文会干扰后面的生成
        entity_by_id = {d["chunk_id"]: d.get("entity_match") for d in reranked}
        off_topic = [c.chunk_id for c in result["selected_chunks"] if entity_by_id.get(c.chunk_id) is False]
        if off_topic:
            warn("上下文里包含没提到查询实体的chunk（可能干扰生成，需要证据评估器把它们挑出来）",
                 f"({len(off_topic)}条: {off_topic})")
        else:
            check("上下文里的chunk都提到了查询实体（或查询没识别出实体）", True)

        print("\n组装元数据:", result["metadata"])
        print("\n--- 组装出来的上下文（人工检查：来源标注、内容是否切题、结尾是否完整）---")
        print(result["context_text"])
        print("--- 上下文结束 ---")

        filled = check_prompts(assembler, q, result["context_text"])
        if "evidence_evaluator" in filled:
            stage = PROMPT_STAGES["evidence_evaluator"]
            print("\n--- 填好的证据评估器提示词（人工检查：这就是将要发给模型的内容）---")
            print("[system]\n" + stage.system_prompt)
            print("\n[user]\n" + filled["evidence_evaluator"])
            print("--- 提示词结束 ---")


if __name__ == "__main__":
    assembler = ContextAssembler(max_context_tokens=MAX_CONTEXT_TOKENS)

    run_offline_tests(assembler)

    if "--offline" not in sys.argv:
        run_real_pipeline_tests(assembler, TEST_QUERIES)

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s in results if s == "PASS")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    n_warn = sum(1 for _, s in results if s == "WARN")
    print(f"汇总：PASS={n_pass}  FAIL={n_fail}  WARN={n_warn}")
    if n_fail:
        print("失败的检查:")
        for name, status in results:
            if status == "FAIL":
                print("  -", name)
    if n_warn:
        print("警告的检查（不一定是错，需要人工看一下）:")
        for name, status in results:
            if status == "WARN":
                print("  -", name)
