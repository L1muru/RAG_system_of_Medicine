"""
完整生成流程测试：真实检索 + 真实本地LLM

把前面所有模块串起来跑一遍：
    查询理解 -> 多路检索 + 融合 -> 重排序 -> 上下文组装 -> 证据评估/筛选 -> 答案草稿 -> 批判性审查 -> 最终答案 -> 后处理

对每个测试问题：
    1. 打印并记录最终答案、参考来源、每个阶段的耗时/状态/token数、答案长度等统计信息
    2. 做几项自动检查（有没有答案、有没有引用编号、编号是否都对得上来源、有没有免责声明）
    3. 日志写到 generation_logs/ 文件夹：
         generation_test_时间.log     完整的文字日志（问题、答案、来源、统计），时间格式是年月日_时分秒
         generation_metrics.jsonl      每次生成的关键指标，一次生成一行（耗时、答案长度、各阶段状态），可以累计比较
       generation_logs/ 在这个脚本所在的目录下，不存在会自动创建；jsonl是追加写入的，不会覆盖以前的记录

运行前需要：
    1. 启动Ollama：ollama serve，并且已经 ollama pull qwen3:8b
    2. 在medrag环境里运行（需要chromadb、sentence-transformers等）

用法：
    python 21_test_generation_pipeline.py                       用内置的4个测试问题跑完整流程
    python 21_test_generation_pipeline.py --query "你的问题"     只测一个自己的问题
    python 21_test_generation_pipeline.py --no-eval --no-review  关掉证据评估和批判性审查（只生成草稿）
    python 21_test_generation_pipeline.py --compare              每个问题跑两遍：只生成草稿 vs 完整四阶段，对比耗时和答案
    python 21_test_generation_pipeline.py --model qwen3:4b       换一个Ollama模型
（--compare 的时候两遍的开关是固定的，--no-eval / --no-review 不起作用；每遍都会重新跑一次检索）
（不加 --compare 时，评估和审查都开着标记为"完整流程"，只要关了一个就标记为"自定义"）
（先连Ollama，连不上就直接返回1，不会去加载检索模型；正常跑完返回0）

注意：模型答案的质量需要人工看日志判断，自动检查只能发现格式和引用方面的问题。
"""

import importlib
import argparse
import json
import logging
import os
import re
import sys
import time

_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
_m_llm_generator = importlib.import_module("19_llm_generator")
LLMGenerator = _m_llm_generator.LLMGenerator
_m_medical_generation_pipeline = importlib.import_module("20_medical_generation_pipeline")
MedicalGenerationPipeline = _m_medical_generation_pipeline.MedicalGenerationPipeline
format_result_report = _m_medical_generation_pipeline.format_result_report
DISCLAIMER_KEYWORD = _m_medical_generation_pipeline.DISCLAIMER_KEYWORD
STAGE_NAMES = _m_medical_generation_pipeline.STAGE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "generation_logs")
METRICS_PATH = os.path.join(LOG_DIR, "generation_metrics.jsonl")

# 检索部分的参数，跟13_retrieval_pipeline.py、16_verify_generation_prep.py保持一致
BM25_MAX_CHUNKS = 20000
TOP_K_VECTOR = 15
TOP_K_KEYWORD = 15
TOP_K_FUSED = 10
FUSION_STRATEGY = "rrf"
MAX_CONTEXT_TOKENS = 2000

# 测试问题。前三个是前面几周一直在用的；第四个是故意问一个知识库里大概率证据不多的问题，
# 看模型是会老实说"证据不足"，还是硬编一个答案出来
TEST_QUERIES = [
    "What effects does metformin have on cardiovascular diseases?",
    "MI patients with HTN, what's the aspirin dosage?",
    "What is the relationship between LDL cholesterol and stroke risk?",
    "What is the recommended antiviral treatment for dengue fever in children?",
]

logger = logging.getLogger("generation_test")


def setup_logging():
    """配置日志：同时输出到屏幕和generation_logs/下带时间的日志文件，返回日志文件路径"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "generation_test_" + time.strftime("%Y%m%d_%H%M%S") + ".log")

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_path


def load_retriever():
    """加载embedding模型、向量索引、BM25和重排序模型，返回retrieve(query)函数，它返回(查询理解结果, 重排序后的候选列表)"""
    # 这些库比较重，放在函数里导入
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

    logger.info("加载embedding模型、连接向量索引、构建BM25索引、加载重排序模型（第一次会比较慢）...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    retriever = MultiPathRetriever(collection, embed_model, max_chunks=BM25_MAX_CHUNKS)
    reranker = Reranker()

    def retrieve(query):
        # 查询理解 -> 多路检索+融合 -> 重排序（不截断，全部候选交给上下文组装器去去重和挑选）
        query_info = process_medical_query(query)
        fused = retriever.retrieve(query_info, top_k_vector=TOP_K_VECTOR, top_k_keyword=TOP_K_KEYWORD,
                                   fusion_strategy=FUSION_STRATEGY, top_k=TOP_K_FUSED)
        reranked = reranker.rerank(query_info, fused, top_k=None)
        return query_info, reranked

    return retrieve


def auto_check(result):
    """对一次生成结果做几项自动检查，返回[(检查名, 状态, 说明)]，状态是PASS/FAIL/WARN"""
    checks = []
    metrics = result["generation_metrics"]
    answer = result["answer"]

    draft_ok = metrics["stage_success"]["answer_generation"]
    checks.append(("生成了答案", "PASS" if draft_ok else "FAIL",
                   "" if draft_ok else "答案生成阶段失败：" + answer))
    if not draft_ok:
        return checks

    checks.append(("带有免责声明", "PASS" if DISCLAIMER_KEYWORD in answer else "FAIL", ""))

    # 只统计正文里的引用编号，不算后面参考文献列表里的 [1] [2]：
    # 在"参考文献/检索到的相关文献/免责声明"这几个段落标题前面切开，只取前半段
    body = re.split(r"\n\n(?:参考文献|检索到的相关文献|免责声明)", answer)[0]
    refs = [int(n) for n in re.findall(r"\[(\d+)\]", body)]
    if not result["sources"]:
        checks.append(("引用编号", "WARN", "没有参考来源（上下文为空）"))
    elif not refs:
        checks.append(("答案里有引用编号", "WARN", "答案没有标注任何引用，无法追溯来源"))
    else:
        bad = [n for n in refs if n < 1 or n > len(result["sources"])]
        checks.append(("引用编号都对得上来源", "PASS" if not bad else "FAIL",
                       f"（共{len(refs)}处引用）" if not bad else f"（超出范围的编号：{bad}）"))

    # citation_check只有走到后处理才会有，所以先判断是不是None
    unmatched = result["intermediate_results"]["citation_check"]["unmatched_citations"] \
        if result["intermediate_results"]["citation_check"] else []
    if unmatched:
        checks.append(("没有对不上来源的引用", "WARN", f"模型引用了不在上下文里的文献：{unmatched}"))

    if metrics["warnings"]:
        checks.append(("流程中没有警告", "WARN", "；".join(metrics["warnings"])))
    return checks


def write_metrics(result, mode):
    # 每次生成追加一行关键指标到jsonl文件
    metrics = result["generation_metrics"]
    tokens = metrics["token_counts"]
    record = {
        "timestamp": result["timestamp"],
        "mode": mode,
        "query": result["query"],
        "total_time_seconds": round(metrics["total_time_seconds"], 2),
        "answer_length_chars": metrics["answer_length_chars"],
        "stage_times": {k: round(v, 2) for k, v in metrics["stage_times"].items()},
        "stage_success": metrics["stage_success"],
        "prompt_tokens": tokens["total_prompt_tokens"],
        "completion_tokens": tokens["total_completion_tokens"],
        "chunks_in_context": result["context_metadata"].get("chunks_in_context"),
        "context_tokens": result["context_metadata"]["estimated_tokens"],
        "num_sources": len(result["sources"]),
        "num_cited": sum(1 for s in result["sources"] if s["cited"]),
        "num_warnings": len(metrics["warnings"]),
    }
    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def log_retrieval(query_info, reranked):
    """把识别出的实体和重排序后前5条候选的分数、实体匹配情况、标题写进日志"""
    logger.info(f"识别实体：{query_info.get('entities')}")
    logger.info(f"重排序后的候选共{len(reranked)}条（前5条）：")
    for i, doc in enumerate(reranked[:5]):
        meta = doc.get("metadata", {})
        logger.info(f"  [{i + 1}] 分数={doc.get('final_score', 0):.3f}  实体匹配={doc.get('entity_match')}"
                    f"  {meta.get('source_title')}")


def run_one(pipeline, retrieve, query, mode_name, enable_eval, enable_review):
    # 跑一个问题，打日志，返回(结果, 指标记录, 自动检查结果)
    logger.info("")
    logger.info("#" * 70)
    logger.info(f"测试问题（模式：{mode_name}）：{query}")

    t = time.time()
    query_info, reranked = retrieve(query)
    retrieval_seconds = time.time() - t
    log_retrieval(query_info, reranked)
    logger.info(f"检索耗时：{retrieval_seconds:.1f}秒（不计入下面的生成耗时）")

    result = pipeline.generate(query, reranked, enable_evaluation=enable_eval, enable_review=enable_review)
    logger.info(format_result_report(result))

    checks = auto_check(result)
    logger.info("")
    logger.info("【自动检查】")
    for name, status, detail in checks:
        logger.info(f"  [{status}] {name} {detail}")

    record = write_metrics(result, mode_name)
    return result, record, checks


def log_summary(records, all_checks):
    """全部问题跑完后，把每次生成的指标列成一张表，再给出平均耗时/答案长度和自动检查的PASS/FAIL/WARN个数"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("【汇总】每次生成的关键指标")
    logger.info(f"{'模式':<10} {'总耗时(秒)':>10} {'答案长度':>8} {'成功阶段':>8} {'引用/来源':>10} {'警告':>4}  问题")
    for r in records:
        succeeded = sum(1 for v in r["stage_success"].values() if v is True)
        ran = sum(1 for v in r["stage_success"].values() if v is not None)
        logger.info(f"{r['mode']:<10} {r['total_time_seconds']:>10.1f} {r['answer_length_chars']:>8} "
                    f"{succeeded:>4}/{ran:<3} {r['num_cited']:>5}/{r['num_sources']:<4} {r['num_warnings']:>4}  "
                    f"{r['query'][:45]}")

    if records:
        avg_time = sum(r["total_time_seconds"] for r in records) / len(records)
        avg_len = sum(r["answer_length_chars"] for r in records) / len(records)
        logger.info(f"平均：总耗时{avg_time:.1f}秒，答案长度{avg_len:.0f}个字符")

    n_pass = sum(1 for checks in all_checks for _, s, _ in checks if s == "PASS")
    n_fail = sum(1 for checks in all_checks for _, s, _ in checks if s == "FAIL")
    n_warn = sum(1 for checks in all_checks for _, s, _ in checks if s == "WARN")
    logger.info(f"自动检查：PASS={n_pass}  FAIL={n_fail}  WARN={n_warn}")


def main():
    """解析命令行参数，先测Ollama连接，再加载检索组件，对每个问题跑生成流程（普通模式或--compare模式），最后打印汇总。返回退出码"""
    parser = argparse.ArgumentParser(description="完整生成流程测试")
    parser.add_argument("--query", help="只测这一个问题")
    parser.add_argument("--no-eval", action="store_true", help="关闭证据评估")
    parser.add_argument("--no-review", action="store_true", help="关闭批判性审查")
    parser.add_argument("--compare", action="store_true", help="每个问题跑两遍：只生成草稿 vs 完整流程，对比")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama模型名")
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info(f"日志文件：{log_path}")
    logger.info(f"指标文件：{METRICS_PATH}")

    # 先测试Ollama，连不上就不用加载那些很慢的检索模型了
    llm = LLMGenerator(model_name=args.model)
    logger.info(llm.connection_message)
    if not llm.available:
        logger.info("请先启动Ollama并下载模型，然后重新运行。")
        return 1

    assembler = ContextAssembler(max_context_tokens=MAX_CONTEXT_TOKENS)
    pipeline = MedicalGenerationPipeline(assembler=assembler, llm=llm, max_context_tokens=MAX_CONTEXT_TOKENS)
    retrieve = load_retriever()

    queries = [args.query] if args.query else TEST_QUERIES
    enable_eval = not args.no_eval
    enable_review = not args.no_review

    records = []
    all_checks = []
    for query in queries:
        if args.compare:
            base_result, base_record, base_checks = run_one(
                pipeline, retrieve, query, "仅草稿", enable_eval=False, enable_review=False)
            full_result, full_record, full_checks = run_one(
                pipeline, retrieve, query, "完整流程", enable_eval=True, enable_review=True)
            records += [base_record, full_record]
            all_checks += [base_checks, full_checks]

            logger.info("")
            logger.info("【对比：仅草稿 vs 完整流程】")
            logger.info(f"  耗时：{base_record['total_time_seconds']:.1f}秒 -> {full_record['total_time_seconds']:.1f}秒")
            logger.info(f"  答案长度：{base_record['answer_length_chars']} -> {full_record['answer_length_chars']}")
            logger.info(f"  引用来源数：{base_record['num_cited']} -> {full_record['num_cited']}")
            same = base_result["answer"] == full_result["answer"]
            logger.info(f"  最终答案是否完全相同：{'是' if same else '否'}（内容差异请对照上面两份答案人工判断）")
        else:
            mode = "完整流程" if (enable_eval and enable_review) else "自定义"
            result, record, checks = run_one(pipeline, retrieve, query, mode, enable_eval, enable_review)
            records.append(record)
            all_checks.append(checks)

    log_summary(records, all_checks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
