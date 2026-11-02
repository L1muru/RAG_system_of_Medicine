"""
答案评估 + 缓存 + 批量处理 的真实测试：真实检索 + 真实本地LLM，用上周的4个测试问题

流程：
    准备   检索：4个问题各检索一次，检索结果后面几个阶段反复用（检索不是这次要测的）
    A      缓存未命中：用 CachedPipeline 逐个生成（缓存是空的，每个都要真的调用模型），
           并用 AnswerEvaluator 评估（ROUGE、关键信息召回、幻觉风险、可读性）
    B      缓存命中：同样的4个问题再来一遍，应该全部命中、答案和A完全一样、几乎不花时间
    C      批量处理：4个问题 + 1个重复问题 + 1个故意写错的任务，用线程池并行生成（缓存是空的），
           检查顺序、失败隔离、重复问题合并，并和A阶段逐个生成的总耗时对比
    D      评估器对照：不调用模型，用几个"已知答案"检查评估器是不是真的有区分度
           （标准答案对自己 / 无关答案 / 夸大答案 / 老实说证据不足）
    E      缓存机制演示：温度过高不缓存、TTL过期、LRU淘汰（用真实流水线 + 小缓存）

重要说明（请一起看结果）：
    1. 下面 REFERENCE_ANSWERS 里的"标准答案"是我根据一般医学知识写的示例，用来演示评估器怎么用，
       不是权威标准答案，也不是从你的语料库里来的。ROUGE和关键信息召回衡量的是"和这段示例像不像"，
       不是"对不对"。想要有意义的评估，请换成医生写的或指南里的标准答案（--reference-file）。
    2. 第4个问题（登革热抗病毒药）的示例标准答案写的是"没有特效抗病毒药"；如果模型老实回答"证据不足"，
       分数会很低，但那其实是诚实的回答——这就是只看相似度的局限。
    3. 批量并行有没有加速，取决于Ollama能不能同时处理多个请求（通常是排队），请看C阶段的实测耗时。

运行前需要：
    1. 启动Ollama：ollama serve，并且已经 ollama pull qwen3:8b
    2. 在medrag环境里运行；另外要 pip install rouge
    3. 检索用的向量索引等要先准备好（检索部分直接用 21_test_generation_pipeline.py 里的 load_retriever，
       会加载embedding模型、向量索引、BM25和重排序模型，第一次会比较慢）

用法：
    python 27_test_evaluation_cache_batch.py                  完整测试（约10~20分钟，A和C各生成4次完整流程）
    python 27_test_evaluation_cache_batch.py --quick          只用前2个问题
    python 27_test_evaluation_cache_batch.py --skip-batch     不做C阶段
    python 27_test_evaluation_cache_batch.py --skip-gating    不做E阶段（缓存机制演示，E1会多调用2次模型）
    python 27_test_evaluation_cache_batch.py --workers 2      C阶段指定线程数（默认自动：不超过CPU核心数，最多2）
    python 27_test_evaluation_cache_batch.py --reference-file refs.json   用自己的标准答案，格式 {"问题": "标准答案"}
    python 27_test_evaluation_cache_batch.py --model qwen3:4b

用自己的 --reference-file 时，测试的问题仍然来自 21_test_generation_pipeline.py 里的 TEST_QUERIES，
而且只保留在文件里有标准答案的那些问题（文件里多出来的问题不会被测试）。

输入：4个内置测试问题（TEST_QUERIES）+ 标准答案（内置示例或 --reference-file）。
输出：
    1. 屏幕和日志文件：generation_logs/eval_cache_batch_test_<时间>.log（目录在 21_test_generation_pipeline.py 旁边）
    2. 每个被评估的答案一行指标，追加到 generation_logs/evaluation_metrics.jsonl
    3. 最后汇总自动检查的 PASS/FAIL 数；Ollama连不上、没有可测试的问题、所有检索都失败时，main()返回1（退出码1），否则返回0
       （有检查FAIL时退出码仍然是0，要看汇总）
"""

import argparse
import json
import os
import sys
import time

import copy
import importlib

t21 = importlib.import_module("21_test_generation_pipeline")
# 文件名以数字开头（23_answer_evaluator.py），不能用 import 语句直接导入，改用 importlib
_evaluator_module = importlib.import_module("23_answer_evaluator")
_batch_module = importlib.import_module("25_batch_generator")
_cache_module = importlib.import_module("24_generation_cache")
AnswerEvaluator = _evaluator_module.AnswerEvaluator
format_evaluation_report = _evaluator_module.format_evaluation_report
compute_rouge = _evaluator_module.compute_rouge
get_answer_body = _evaluator_module.get_answer_body
HAS_ROUGE_LIB = _evaluator_module.HAS_ROUGE_LIB
batch_generate = _batch_module.batch_generate
format_batch_summary = _batch_module.format_batch_summary
CachedPipeline = _cache_module.CachedPipeline
GenerationCache = _cache_module.GenerationCache
format_cache_stats = _cache_module.format_cache_stats
_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
_m_llm_generator = importlib.import_module("19_llm_generator")
LLMGenerator = _m_llm_generator.LLMGenerator
_m_medical_generation_pipeline = importlib.import_module("20_medical_generation_pipeline")
MedicalGenerationPipeline = _m_medical_generation_pipeline.MedicalGenerationPipeline
format_result_report = _m_medical_generation_pipeline.format_result_report
NO_EVIDENCE_ANSWER = _m_medical_generation_pipeline.NO_EVIDENCE_ANSWER
_m_prompt_templates = importlib.import_module("15_prompt_templates")
PROMPT_STAGES = _m_prompt_templates.PROMPT_STAGES

LOG_DIR = t21.LOG_DIR
EVAL_METRICS_PATH = os.path.join(LOG_DIR, "evaluation_metrics.jsonl")
logger = t21.logger          # 和21_test_generation_pipeline共用同一个logger（load_retriever里也往里写日志）

# ---------------------------------------------------------------------------
# 示例标准答案（英文，由我根据一般医学知识写的，只用来演示评估器，不是权威答案，也不是来自语料库）
# 里面故意包含了百分比、剂量、时间范围、安全信息、治疗建议、作用机制，评估器才有东西可比
# ---------------------------------------------------------------------------
REFERENCE_ANSWERS = {
    "What effects does metformin have on cardiovascular diseases?": (
        "In patients with type 2 diabetes, metformin is associated with a lower risk of cardiovascular events. "
        "In the UKPDS 34 trial in overweight patients, metformin reduced myocardial infarction by about 39% and "
        "all-cause mortality by about 36% compared with conventional treatment over roughly 10 years. "
        "Observational studies also suggest a lower risk, but evidence from randomized trials is limited. "
        "Proposed mechanisms include AMPK activation, improved insulin sensitivity, reduced inflammation and weight "
        "benefits. Metformin is recommended as first-line treatment for type 2 diabetes. Common side effects are "
        "gastrointestinal upset, and there is a rare risk of lactic acidosis, so it is contraindicated in severe "
        "renal impairment."),
    "MI patients with HTN, what's the aspirin dosage?": (
        "For secondary prevention after myocardial infarction, low-dose aspirin 75-100 mg once daily is "
        "recommended long term. An initial loading dose of 150-300 mg is given at the acute event. When combined "
        "with a P2Y12 inhibitor as dual antiplatelet therapy, treatment is generally continued for 12 months, "
        "followed by aspirin alone. Blood pressure should be controlled because uncontrolled hypertension "
        "increases the risk of bleeding, especially intracranial hemorrhage. The main risks are gastrointestinal "
        "bleeding and hemorrhagic stroke, and a proton pump inhibitor may be considered for patients at high "
        "gastrointestinal risk. Aspirin works by irreversibly inhibiting cyclooxygenase-1 and thromboxane A2 "
        "production, which reduces platelet aggregation."),
    "What is the relationship between LDL cholesterol and stroke risk?": (
        "Higher LDL cholesterol is associated with a higher risk of ischemic stroke, mainly through "
        "atherosclerosis of large arteries. In statin trials, each 1 mmol/L reduction in LDL cholesterol lowers "
        "the risk of ischemic stroke by roughly 15-25%. Lowering LDL with statins, ezetimibe or PCSK9 inhibitors "
        "is recommended for patients with atherosclerotic disease, usually as long-term treatment. The "
        "relationship with hemorrhagic stroke is less clear, and very low LDL levels have been linked to a possible "
        "small increase in intracerebral hemorrhage, but the evidence is inconsistent. Statin side effects "
        "include muscle symptoms and a small increase in the risk of new-onset diabetes."),
    "What is the recommended antiviral treatment for dengue fever in children?": (
        "There is no specific antiviral treatment for dengue fever, and no antiviral drug is recommended for "
        "children. Treatment is supportive: oral rehydration and fluid replacement, rest, and close monitoring "
        "for warning signs such as persistent vomiting, abdominal pain, mucosal bleeding and lethargy. "
        "Paracetamol 10-15 mg/kg per dose, not more than 4 doses per 24 hours, is recommended for fever and "
        "pain. Aspirin and NSAIDs such as ibuprofen should be avoided because of the risk of bleeding. Children "
        "with warning signs need hospital admission and intravenous fluids, because complications include severe "
        "bleeding, plasma leakage and shock."),
}

# 用来检验评估器的"夸大答案"：故意写成充满绝对化表述
OVERCLAIM_ANSWER = ("研究表明二甲双胍对心血管疾病100%有效，已被证明完全安全，没有任何副作用，所有糖尿病患者都应该立刻服用。")
# 无关答案
UNRELATED_ANSWER = ("斑马鱼的毛细胞在受损之后可以通过支持细胞的增殖再生，这个过程和Notch信号通路有关，"
                    "研究人员用荧光显微镜观察了再生的时间进程。")

eval_log_records = []     # 所有被评估的答案的指标，最后也写进jsonl
check_results = []        # [(检查名, PASS/FAIL, 说明)]


def setup_logging():
    """日志同时写到文件（带时间）和屏幕。用的是21里的那个logger，所以清掉它原来的handler再加，避免重复输出。返回日志文件路径"""
    os.makedirs(LOG_DIR, exist_ok=True)
    import logging
    log_path = os.path.join(LOG_DIR, "eval_cache_batch_test_" + time.strftime("%Y%m%d_%H%M%S") + ".log")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_path


def check(name, passed, detail=""):
    """记录并打印一项自动检查的结果（PASS/FAIL），最后 final_summary 会汇总"""
    status = "PASS" if passed else "FAIL"
    check_results.append((name, status, detail))
    logger.info(f"  [{status}] {name} {detail}")


def short(text, n=40):
    """文字太长就截到n个字符再加省略号，日志里显示问题用"""
    return text if len(text) <= n else text[:n] + "..."


def record_evaluation(phase, label, query, report, result=None, extra=None):
    # 把一次评估的关键指标追加到jsonl，同时放进内存列表供最后汇总
    # phase是阶段（A/C/D），label是这一行的名字；result是生成结果，D阶段没有生成所以是None（耗时和cache_hit记None）；
    # extra是额外要记的字段（C阶段用来记和A阶段答案的相似度）。返回这条记录
    sim = report["similarity"]
    key = report["key_info"]
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase, "label": label, "query": query,
        "body_chars": report["body_chars"],
        "rouge_1_f": sim["rouge-1"]["f"] if sim else None,
        "rouge_2_f": sim["rouge-2"]["f"] if sim else None,
        "rouge_l_f": sim["rouge-l"]["f"] if sim else None,
        "rouge_backend": sim["backend"] if sim else None,
        "key_info_recall": key["overall_recall"] if key else None,
        "key_info_per_category": {k: v["recall"] for k, v in key["per_category"].items()} if key else None,
        "risk_score": report["hallucination"]["risk_score"],
        "risk_level": report["hallucination"]["risk_level"],
        "risk_signals": [s["name"] for s in report["hallucination"]["signals"]],
        "citation_markers": report["hallucination"]["citation_markers"],
        "readability_score": report["readability"]["readability_score"],
        "avg_sentence_length": report["readability"]["avg_sentence_length"],
        "generation_seconds": result["generation_metrics"]["total_time_seconds"] if result else None,
        "cache_hit": bool((result or {}).get("cache", {}).get("hit")) if result else None,
    }
    if extra:
        record.update(extra)
    eval_log_records.append(record)
    with open(EVAL_METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def f_or_dash(value, fmt="{:.3f}"):
    """数值按格式显示，是None（没有标准答案算不出来）就显示"-" """
    return "-" if value is None else fmt.format(value)


def log_eval_table(title, rows):
    """把一组评估记录打印成一张对比表（每个答案一行：ROUGE、关键信息召回、幻觉风险、可读性、耗时）"""
    # rows是 record_evaluation 返回的记录列表
    logger.info("")
    logger.info(title)
    logger.info(f"{'标签':<14} {'R-1':>6} {'R-2':>6} {'R-L':>6} {'关键信息召回':>10} {'幻觉风险':>8} {'可读性':>6} {'耗时(秒)':>8}  问题")
    for r in rows:
        logger.info(f"{r['label']:<14} {f_or_dash(r['rouge_1_f']):>6} {f_or_dash(r['rouge_2_f']):>6} "
                    f"{f_or_dash(r['rouge_l_f']):>6} {f_or_dash(r['key_info_recall'], '{:.2f}'):>10} "
                    f"{r['risk_level'] + '(' + format(r['risk_score'], '.2f') + ')':>8} "
                    f"{r['readability_score']:>6.2f} {f_or_dash(r['generation_seconds'], '{:.1f}'):>8}  {short(r['query'], 38)}")


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------
def phase_prepare(retrieve, queries):
    """准备阶段：每个问题检索一次，返回 {问题: 重排序后的候选}。检索失败的问题跳过。
    后面A~E阶段共用这份检索结果，这样每次生成的输入是一样的，对比才公平"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("【准备】检索（只做一次，后面阶段共用）")
    docs_by_query = {}
    for query in queries:
        t = time.time()
        try:
            query_info, reranked = retrieve(query)
        except Exception as e:
            logger.info(f"  检索失败，跳过这个问题：{short(query, 50)}  原因：{type(e).__name__}: {e}")
            continue
        docs_by_query[query] = reranked
        logger.info(f"  {short(query, 55)}  -> {len(reranked)}条候选，{time.time() - t:.1f}秒")
    return docs_by_query


def phase_a(cached, evaluator, docs_by_query, references):
    """A阶段：缓存是空的，逐个问题真的调用模型生成，并用评估器评估。
    返回 (各问题的生成结果, 这些生成花的总秒数)，总秒数留给C阶段对比批量的耗时"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("【A】缓存未命中：逐个生成 + 评估（缓存是空的，每个问题都要真的调用模型）")
    results = {}
    rows = []
    for query, docs in docs_by_query.items():
        logger.info("")
        logger.info("#" * 70)
        logger.info(f"问题：{query}")
        result = cached.generate(query, docs)
        results[query] = result
        logger.info(format_result_report(result, show_intermediate=False))
        logger.info("")
        logger.info("【给模型的上下文】（组装后；评估器用它核对答案里的数值）")
        logger.info(result["intermediate_results"]["context_text"])
        logger.info("")
        logger.info(f"缓存：{result['cache']}")

        report = evaluator.evaluate_result(result, reference=references.get(query))
        logger.info("")
        logger.info(format_evaluation_report(report, "答案评估"))
        rows.append(record_evaluation("A", "A-生成", query, report, result))

    log_eval_table("【A汇总】", rows)
    first_pass_seconds = sum(r["generation_metrics"]["total_time_seconds"] for r in results.values())
    logger.info(f"A阶段逐个生成的总耗时：{first_pass_seconds:.1f}秒（C阶段批量处理拿它做对比）")
    return results, first_pass_seconds


def phase_b(cached, docs_by_query, results_a):
    """B阶段：用同一个cached把A阶段的问题再问一遍，检查是不是全部命中、答案是不是和A完全一样，并看命中花了多少时间"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("【B】缓存命中：同样的问题再来一遍")
    all_hit = True
    all_same = True
    for query, docs in docs_by_query.items():
        if query not in results_a:
            continue
        t = time.time()
        result = cached.generate(query, docs)
        seconds = time.time() - t
        hit = result["cache"]["hit"]
        same = result["answer"] == results_a[query]["answer"]
        all_hit = all_hit and hit
        all_same = all_same and same
        original = result["cache"].get("original_time_seconds", 0)
        logger.info(f"  {short(query, 50)}  命中={hit}  答案与A完全相同={same}  "
                    f"本次耗时{seconds:.3f}秒（A阶段生成花了{results_a[query]['generation_metrics']['total_time_seconds']:.1f}秒；"
                    f"缓存里记录的原耗时{original:.1f}秒，缓存年龄{result['cache'].get('age_seconds', 0):.1f}秒）")
    logger.info(f"  {format_cache_stats(cached.stats())}")
    check("B：全部命中缓存", all_hit)
    check("B：命中的答案和第一次完全一致", all_same)


def phase_c(model_args, docs_by_query, results_a, first_pass_seconds, evaluator, references, workers, llm, assembler):
    """
    C阶段：用线程池批量生成（全部问题 + 1个重复问题 + 1个格式错误的任务），用的是新缓存。
    检查输出条数/顺序、坏任务不影响别的、重复问题来自缓存，并拿总耗时和A阶段对比，
    再看同一个问题在A和C两次真实生成的答案有多像。参数 workers 是线程数（None表示自动）；
    model_args 在函数里没有用到。
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("【C】批量处理（线程池）：缓存是新的，所以每个问题还是要真的生成一次")
    queries = list(docs_by_query.keys())
    # 新缓存 + 新的CachedPipeline，共用同一个llm和assembler
    pipeline = MedicalGenerationPipeline(assembler=assembler, llm=llm, max_context_tokens=t21.MAX_CONTEXT_TOKENS)
    cached = CachedPipeline(pipeline, cache=GenerationCache())

    # 任务：全部问题 + 第一个问题重复一次 + 一个故意写错的任务（None）。
    # 重复问题和原问题的缓存键一样，同时跑时后到的会等先到的；格式错误的任务放最后，方便按位置检查
    tasks = [(q, docs_by_query[q]) for q in queries]
    tasks.append((queries[0], docs_by_query[queries[0]]))      # 重复的问题
    tasks.append(None)                                         # 格式错误的任务
    logger.info(f"  共{len(tasks)}个任务：{len(queries)}个不同问题 + 1个重复问题（第1个问题）+ 1个格式错误的任务")

    # progress回调是按任务顺序依次调用的（见25），所以日志里的 [n/总数] 是按顺序出来的，不是谁先跑完谁先打印
    def progress(done, total, record):
        status = "成功" if record["success"] else "失败"
        logger.info(f"    [{done}/{total}] #{record['index']} {status}  {record['elapsed']:.1f}秒  "
                    f"缓存命中={record['cache_hit']}  {short(str(record['query']), 45)}"
                    + (f"  错误：{record['error']}" if record["error"] else ""))

    out = batch_generate(cached, tasks, max_workers=workers, progress=progress)
    summary = out["summary"]
    logger.info("  " + format_batch_summary(summary))
    logger.info("  " + format_cache_stats(cached.stats()))
    logger.info(f"  对比：A阶段逐个生成{len(results_a)}个问题共{first_pass_seconds:.1f}秒；"
                f"批量(线程数={summary['workers']})处理{len(queries)}个不同问题"
                f"{'(另有1个重复问题合并)' if len(tasks) > len(queries) + 1 else ''}用了{summary['wall_seconds']:.1f}秒")
    if summary["workers"] > 1:
        ratio = summary["wall_seconds"] / first_pass_seconds if first_pass_seconds else 0
        logger.info(f"  批量耗时是逐个生成的 {ratio:.0%}。如果接近100%，说明Ollama把请求排队处理了，"
                    f"多线程在这台机器上基本没有加速；这个结果是一次测量，有波动。")

    records = out["records"]
    check("C：输出条数和输入一致", len(records) == len(tasks))
    check("C：输出顺序和输入一致", [r["index"] for r in records] == list(range(len(tasks)))
          and all(records[i]["query"] == queries[i] for i in range(len(queries)))
          and records[len(queries)]["query"] == queries[0])
    check("C：格式错误的任务失败，其他任务不受影响",
          (not records[-1]["success"]) and all(r["success"] for r in records[:-1]),
          f"(成功{summary['succeeded']}，失败{summary['failed']})")
    # 重复问题和原问题谁先跑完不一定，所以只要其中一个是命中的，就说明第二份是从缓存来的
    check("C：重复的问题没有重复生成（来自缓存）", records[len(queries)]["cache_hit"] or records[0]["cache_hit"],
          f"(重复问题命中={records[len(queries)]['cache_hit']}，原问题命中={records[0]['cache_hit']})")
    check("C：重复问题的两份答案一致",
          records[0]["result"] is not None and records[0]["result"]["answer"] == records[len(queries)]["result"]["answer"])

    # 低温度不等于确定：对比A阶段和C阶段同一个问题的答案
    logger.info("")
    logger.info("  A阶段 vs C阶段：同一个问题、同样的上下文和设置，两次真实生成的答案有多像（ROUGE-L F1，1.0=完全一样）")
    rows = []
    for i, query in enumerate(queries):
        rec = records[i]
        if not rec["success"] or query not in results_a:
            continue
        body_a = get_answer_body(results_a[query]["answer"])
        body_c = get_answer_body(rec["result"]["answer"])
        rouge = compute_rouge(body_c, body_a)
        same_context = (results_a[query]["intermediate_results"]["context_text"]
                        == rec["result"]["intermediate_results"]["context_text"])
        logger.info(f"    {short(query, 45)}  相同上下文={same_context}  ROUGE-L F1={rouge['rouge-l']['f']:.3f}"
                    f"  答案是否完全相同={body_a == body_c}")
        report = evaluator.evaluate_result(rec["result"], reference=references.get(query))
        rows.append(record_evaluation("C", "C-批量", query, report, rec["result"],
                                      extra={"rouge_l_vs_A": rouge["rouge-l"]["f"]}))
    logger.info("  （如果不是1.0，说明温度0.1~0.3也不是完全确定的；缓存保证的是'命中时返回同一个答案'，"
                "而不是'重新生成会得到同一个答案'。）")
    log_eval_table("【C汇总】批量生成的答案评估", rows)


def phase_d(evaluator, results_a, references):
    """
    D阶段（不调用模型）：同一个标准答案（第1个问题，二甲双胍）下，比较 标准答案自己 / 模型真实答案（有的话）/
    无关答案 / 夸大答案 / "证据不足" 的评估结果，看评估器有没有区分度。
    没有第1个问题的标准答案就整个跳过。
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("【D】评估器对照（不调用模型）：用已知答案检验评估器有没有区分度")
    query = "What effects does metformin have on cardiovascular diseases?"
    reference = references.get(query)
    if reference is None:
        logger.info("  第1个测试问题没有标准答案，跳过D阶段")
        return

    rows = []
    cases = [("标准答案自己", reference), ("无关答案", UNRELATED_ANSWER), ("夸大答案", OVERCLAIM_ANSWER),
             ("证据不足", NO_EVIDENCE_ANSWER)]
    if query in results_a:
        cases.insert(1, ("模型真实答案", results_a[query]["answer"]))
    reports = {}
    for label, answer in cases:
        report = evaluator.evaluate(answer, reference=reference)
        reports[label] = report
        rows.append(record_evaluation("D", label, query, report))
    log_eval_table("【D汇总】同一个标准答案（二甲双胍）下的几种答案", rows)
    logger.info("")
    logger.info(format_evaluation_report(reports["夸大答案"], "夸大答案的详细评估（看触发了哪些幻觉信号）"))

    check("D：标准答案对自己：ROUGE-L=1、关键信息召回=1",
          reports["标准答案自己"]["similarity"]["rouge-l"]["f"] > 0.999 and reports["标准答案自己"]["key_info"]["overall_recall"] == 1.0)
    check("D：无关答案的ROUGE-L和关键信息召回都明显低于标准答案自己",
          reports["无关答案"]["similarity"]["rouge-l"]["f"] < 0.3 and reports["无关答案"]["key_info"]["overall_recall"] < 0.3,
          f"(R-L={reports['无关答案']['similarity']['rouge-l']['f']:.2f}，召回={reports['无关答案']['key_info']['overall_recall']:.2f})")
    check("D：夸大答案被判为高幻觉风险", reports["夸大答案"]["hallucination"]["risk_level"] == "高",
          f"(风险分={reports['夸大答案']['hallucination']['risk_score']:.2f}，信号{reports['夸大答案']['hallucination']['signal_count']}个)")
    check("D：标准答案自己的幻觉风险是低", reports["标准答案自己"]["hallucination"]["risk_level"] == "低")
    if "模型真实答案" in reports:
        # 这一项只记录，不做通过/失败判断：模型答案如果是中文而标准答案是英文，ROUGE本来就接近0，比较没有意义
        real = reports["模型真实答案"]
        logger.info(f"  模型真实答案：ROUGE-L={real['similarity']['rouge-l']['f']:.3f}，关键信息召回="
                    f"{f_or_dash(real['key_info']['overall_recall'], '{:.2f}')}，幻觉风险={real['hallucination']['risk_level']}"
                    f"（仅供参考，不做通过/失败判断）")
    logger.info("  注意'证据不足'这一行：它是一个诚实的回答，但ROUGE和关键信息召回都很低——相似度指标不能判断诚实与否。")


def phase_e(llm, assembler, docs_by_query):
    """
    E阶段：缓存机制演示，用第一个问题。
    E1 把答案生成器温度改成0.8，检查不会被缓存（会真的调用模型2次）；
    E2/E3 把E1的结果放进小缓存，检查TTL过期、LRU淘汰、字节上限（不再调用模型）。
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("【E】缓存机制演示")
    query = next(iter(docs_by_query))
    docs = docs_by_query[query]

    # E1 温度限制：把答案生成器的温度提到0.8，只跑草稿阶段（省时间）
    logger.info("")
    logger.info("E1 温度限制：答案生成器温度改成0.8（只跑草稿，不评估不审查，省时间）")
    hot_stages = copy.deepcopy(PROMPT_STAGES)
    hot_stages["answer_generator"].temperature = 0.8
    hot_pipeline = MedicalGenerationPipeline(assembler=assembler, llm=llm, prompt_stages=hot_stages,
                                             max_context_tokens=t21.MAX_CONTEXT_TOKENS)
    hot = CachedPipeline(hot_pipeline, cache=GenerationCache())
    r1 = hot.generate(query, docs, enable_evaluation=False, enable_review=False)
    r2 = hot.generate(query, docs, enable_evaluation=False, enable_review=False)
    logger.info(f"  第1次：{r1['cache']}")
    logger.info(f"  第2次：{r2['cache']}")
    logger.info(f"  两次答案是否完全相同：{get_answer_body(r1['answer']) == get_answer_body(r2['answer'])}"
                f"（温度高时通常不同，所以不缓存是合理的）")
    check("E1：温度0.8的结果没有被缓存，第二次重新生成", (not r1["cache"]["stored"]) and (not r2["cache"]["hit"]) and len(hot.cache) == 0)

    # E2 TTL 和 LRU：用真实流水线的结果放进小缓存，不需要再调用模型
    logger.info("")
    logger.info("E2 TTL过期 和 E3 LRU淘汰（把一份真实生成结果放进小缓存，不再调用模型）")
    # 直接拿E1的结果当样本。把答案生成阶段标成成功，是为了让它通过cache.put的"结果是否成功"检查
    # （样本被改过，只在这里的演示里用）
    sample = r1
    sample["generation_metrics"]["stage_success"]["answer_generation"] = True
    ttl_cache = GenerationCache(ttl_seconds=1.0)
    ttl_cache.put("k", sample, 0.1)
    hit_before = ttl_cache.get("k")[0] is not None
    time.sleep(1.2)
    hit_after = ttl_cache.get("k")[0] is not None
    logger.info(f"  TTL=1秒：存入后立即取 命中={hit_before}；等1.2秒后再取 命中={hit_after}")
    check("E2：TTL过期后取不到", hit_before and not hit_after)

    lru = GenerationCache(max_entries=2)
    lru.put("k1", sample, 0.1)
    lru.put("k2", sample, 0.1)
    lru.get("k1")
    lru.put("k3", sample, 0.1)
    logger.info(f"  最多2条：存k1、k2，取k1，再存k3 -> 还在的键：{[k for k in ('k1', 'k2', 'k3') if k in lru]}（k2最久没用，被淘汰）")
    check("E3：LRU淘汰最久没用的", "k2" not in lru and "k1" in lru and "k3" in lru)

    # 条数上限放大到100，让淘汰只可能是字节上限触发的
    size_cache = GenerationCache(max_entries=100, max_bytes=40 * 1024)
    for i in range(20):
        size_cache.put(f"s{i}", sample, 0.1)
    stats = size_cache.stats()
    logger.info(f"  字节上限40KB：连续存20份结果后 {stats['entries']}条 / {stats['total_bytes'] / 1024:.1f}KB，淘汰{stats['evictions']}次")
    check("E3：总字节数不超过上限", stats["total_bytes"] <= 40 * 1024)


def final_summary():
    """打印所有自动检查的 PASS/FAIL 数量，失败的逐条列出，并说明评估指标文件的位置"""
    logger.info("")
    logger.info("=" * 70)
    n_pass = sum(1 for _, s, _ in check_results if s == "PASS")
    n_fail = sum(1 for _, s, _ in check_results if s == "FAIL")
    logger.info(f"自动检查汇总：PASS={n_pass}  FAIL={n_fail}")
    for name, status, detail in check_results:
        if status == "FAIL":
            logger.info(f"  [FAIL] {name} {detail}")
    logger.info(f"评估指标已追加到：{EVAL_METRICS_PATH}（每个被评估的答案一行）")


def load_references(path):
    """读标准答案：不给路径就用内置的示例（REFERENCE_ANSWERS）；给了就读JSON文件，格式 {问题: 标准答案}"""
    if not path:
        return dict(REFERENCE_ANSWERS)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    """解析命令行参数，检查Ollama，加载检索器，按 准备 -> A -> B -> C -> D -> E 的顺序跑，最后汇总。成功返回0，前置条件不满足返回1"""
    parser = argparse.ArgumentParser(description="答案评估 + 缓存 + 批量处理 真实测试")
    parser.add_argument("--quick", action="store_true", help="只用前2个测试问题")
    parser.add_argument("--skip-batch", action="store_true", help="不做C阶段（批量处理）")
    parser.add_argument("--skip-gating", action="store_true", help="不做E阶段（缓存机制演示，会多调用2次模型）")
    parser.add_argument("--workers", type=int, default=None, help="C阶段线程数（默认自动）")
    parser.add_argument("--reference-file", help="JSON文件：{问题: 标准答案}，不给就用内置的示例标准答案")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama模型名")
    args = parser.parse_args()

    log_path = setup_logging()
    logger.info(f"日志文件：{log_path}")
    if not HAS_ROUGE_LIB:
        logger.info("提示：没有安装rouge库，ROUGE用的是自己写的简化版（pip install rouge 之后再运行可用库）")
    if not args.reference_file:
        logger.info("提示：使用的是内置的示例标准答案（我根据一般医学知识写的，不是权威答案，也不是来自语料库），"
                    "分数只表示和这段示例的相似程度。")

    llm = LLMGenerator(model_name=args.model)
    logger.info(llm.connection_message)
    if not llm.available:
        logger.info("请先启动Ollama并下载模型，然后重新运行。")
        return 1

    references = load_references(args.reference_file)
    queries = t21.TEST_QUERIES[:2] if args.quick else list(t21.TEST_QUERIES)
    # 用自己的标准答案文件时，只测文件里有标准答案的问题；用内置的就不用过滤
    queries = [q for q in queries if q in references] if args.reference_file else queries
    if not queries:
        logger.info("没有可测试的问题（标准答案文件里的问题和内置测试问题对不上）。")
        return 1

    # 这个assembler之后也会交给C、E阶段新建的pipeline共用（只有缓存是各自新的）
    assembler = ContextAssembler(max_context_tokens=t21.MAX_CONTEXT_TOKENS)
    pipeline = MedicalGenerationPipeline(assembler=assembler, llm=llm, max_context_tokens=t21.MAX_CONTEXT_TOKENS)
    cached = CachedPipeline(pipeline, cache=GenerationCache())
    evaluator = AnswerEvaluator()
    retrieve = t21.load_retriever()

    docs_by_query = phase_prepare(retrieve, queries)
    if not docs_by_query:
        logger.info("所有问题的检索都失败了。")
        return 1

    results_a, first_pass_seconds = phase_a(cached, evaluator, docs_by_query, references)
    phase_b(cached, docs_by_query, results_a)
    if not args.skip_batch:
        phase_c(args, docs_by_query, results_a, first_pass_seconds, evaluator, references, args.workers, llm, assembler)
    phase_d(evaluator, results_a, references)
    if not args.skip_gating:
        phase_e(llm, assembler, docs_by_query)

    final_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
