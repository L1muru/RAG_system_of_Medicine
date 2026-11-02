"""
离线验证脚本：检查 答案评估器（23_answer_evaluator.py）、生成缓存（24_generation_cache.py）、
批量处理（25_batch_generator.py）的逻辑对不对

不需要启动Ollama，也不需要加载任何模型：
    - 评估器：给定输入，和手算的期望结果对比；ROUGE还会拿rouge库和自己写的简化版互相核对
    - 缓存：用假时钟测过期，用假的Ollama（22_verify_generation_logic.py里的ScriptedLLM）测命中/不命中
    - 批量：用假的生成器控制每个任务的耗时和失败，检查顺序、失败隔离、线程数

用法：python 26_verify_eval_cache_batch.py
（要先 pip install rouge，没装的话ROUGE库核对那几项会跳过）

没有命令行参数。要在这些脚本所在的目录里运行（用 importlib 按文件名导入 14_、15_、20_、22_、23_、24_、25_ 这几个文件）。
输入：脚本里写死的测试数据。输出：每项检查打印一行 [PASS]/[FAIL]，最后打印 PASS/FAIL 的数量，
失败的检查会再列一遍。注意：代码里没有 sys.exit，所以有检查失败时退出码也是0，要看打印出来的汇总。

测试分三部分（每个 test_ 函数对应下面的"测试N"）：
    一、评估器  测试1~5      二、缓存  测试6~12      三、批量处理  测试13~15

注意：这个脚本只能验证"代码逻辑"。评估指标在真实答案上的表现、批量并行在真实Ollama上有没有加速，
要用 27_test_evaluation_cache_batch.py 看。
"""

import copy
import importlib
import threading
import time

# 文件名以数字开头（23_answer_evaluator.py），不能用 import 语句直接导入，改用 importlib
ev = importlib.import_module("23_answer_evaluator")
_cache_module = importlib.import_module("24_generation_cache")
_batch_module = importlib.import_module("25_batch_generator")
AnswerEvaluator = ev.AnswerEvaluator
batch_generate = _batch_module.batch_generate
choose_worker_count = _batch_module.choose_worker_count
format_batch_summary = _batch_module.format_batch_summary
CachedPipeline = _cache_module.CachedPipeline
GenerationCache = _cache_module.GenerationCache
make_cache_key = _cache_module.make_cache_key
format_cache_stats = _cache_module.format_cache_stats
_m_prompt_templates = importlib.import_module("15_prompt_templates")
PROMPT_STAGES = _m_prompt_templates.PROMPT_STAGES
_m_verify_generation_logic = importlib.import_module("22_verify_generation_logic")
ScriptedLLM = _m_verify_generation_logic.ScriptedLLM
FULL_SCRIPT = _m_verify_generation_logic.FULL_SCRIPT
make_docs = _m_verify_generation_logic.make_docs
make_pipeline = _m_verify_generation_logic.make_pipeline
TITLE_1 = _m_verify_generation_logic.TITLE_1

results = []       # [(检查名, PASS/FAIL)]，最后汇总用


def check(name, passed, detail=""):
    """记录并打印一项检查的结果（PASS/FAIL）"""
    status = "PASS" if passed else "FAIL"
    results.append((name, status))
    print(f"  [{status}] {name} {detail}")


def close(a, b, tol=1e-6):
    """两个小数是不是差不多相等（浮点数不能直接用==比）"""
    return abs(a - b) <= tol


# ===========================================================================
# 一、评估器
# ===========================================================================
def test_rouge():
    """检查ROUGE：自写版的手算结果、方向（召回/精确）、空输入、语言不一致提醒、引用标记、中文按字切分，以及和rouge库的结果互相核对"""
    print("\n--- 测试1：ROUGE ---")
    same = ev.compute_rouge("alpha beta gamma", "alpha beta gamma", backend="builtin")
    check("builtin：完全相同 -> 三个F1都是1", all(close(same[k]["f"], 1.0) for k in ("rouge-1", "rouge-2", "rouge-l")))
    diff = ev.compute_rouge("alpha beta", "gamma delta", backend="builtin")
    check("builtin：完全不同 -> 都是0", all(diff[k]["f"] == 0 for k in ("rouge-1", "rouge-2", "rouge-l")))

    # 手算：hyp = a b c d，ref = a b x d
    # rouge-1: 重叠3，p=r=0.75；rouge-2: hyp的二元组 ab bc cd，ref的 ab bx xd，重叠1，p=r=1/3；rouge-l: LCS = a b d = 3 -> 0.75
    hand = ev.compute_rouge("alpha beta cc dd", "alpha beta xx dd", backend="builtin")
    check("builtin：手算的ROUGE-1=0.75", close(hand["rouge-1"]["r"], 0.75) and close(hand["rouge-1"]["p"], 0.75))
    check("builtin：手算的ROUGE-2=1/3", close(hand["rouge-2"]["r"], 1 / 3))
    check("builtin：手算的ROUGE-L=0.75", close(hand["rouge-l"]["r"], 0.75))

    # 召回和精确率方向：hyp比ref短
    short = ev.compute_rouge("alpha beta", "alpha beta gamma delta", backend="builtin")
    check("召回=重叠/标准答案词数，精确=重叠/生成词数", close(short["rouge-1"]["r"], 0.5) and close(short["rouge-1"]["p"], 1.0))

    # 空输入不报错
    for hyp, ref, label in (("", "alpha", "生成为空"), ("alpha", "", "标准为空"), ("", "", "都为空"), ("。，！", "alpha", "只有标点")):
        r = ev.compute_rouge(hyp, ref)
        check(f"空输入不报错且为0：{label}", r["rouge-1"]["f"] == 0 and r["rouge-l"]["f"] == 0)

    # 语言不一致提醒
    mixed = ev.compute_rouge("二甲双胍降低风险", "metformin lowers risk", backend="builtin")
    same_lang = ev.compute_rouge("metformin lowers risk", "metformin reduces risk", backend="builtin")
    both_zh = ev.compute_rouge("二甲双胍降低风险", "二甲双胍减少风险", backend="builtin")
    half = ev.compute_rouge("metformin 降低 risk factor", "metformin lowers risk", backend="builtin")
    check("语言不一致(中文对英文)：language_mismatch=True，ROUGE接近0", mixed["language_mismatch"] and mixed["rouge-1"]["f"] == 0)
    check("同语言(英对英、中对中)：不提醒", not same_lang["language_mismatch"] and not both_zh["language_mismatch"])
    check("中英混合的答案对英文标准答案：占比差不到0.6，不提醒", not half["language_mismatch"], f"({half['hyp_cjk_ratio']:.2f} vs {half['ref_cjk_ratio']:.2f})")
    check("空输入也带语言字段", "language_mismatch" in ev.compute_rouge("", "abc"))
    report = ev.AnswerEvaluator(rouge_backend="builtin").evaluate("二甲双胍降低风险", reference="metformin lowers risk")
    check("报告里显示语言不一致的提醒", "语言差别很大" in ev.format_evaluation_report(report))

    # 引用标记不影响
    with_cite = ev.compute_rouge("alpha beta [1] gamma [来源: Some Paper]", "alpha beta gamma", backend="builtin")
    check("引用标记[1]、[来源: ...]被去掉后再算", close(with_cite["rouge-1"]["f"], 1.0))

    # 中文按字切分
    zh = ev.compute_rouge("二甲双胍降低血糖", "二甲双胍降低血压", backend="builtin")
    check("中文按单个汉字算：8个字里7个相同", close(zh["rouge-1"]["r"], 7 / 8))

    # 库和自己写的互相核对：两边结果一致，才说明自写版是对的（没装库就跳过，直接return）
    if not ev.HAS_ROUGE_LIB:
        print("  [SKIP] 没有安装rouge库，跳过库和builtin的互相核对（pip install rouge）")
        return
    pairs = [
        ("metformin reduces cardiovascular events in diabetes patients",
         "metformin is associated with reduced cardiovascular events in patients with type 2 diabetes"),
        ("二甲双胍可以降低2型糖尿病患者的心血管事件风险", "二甲双胍与2型糖尿病患者较少的心血管事件相关"),
        ("aspirin 75 mg daily for 12 months", "aspirin 100 mg once daily for 12 months after stenting"),
        ("alpha alpha beta", "alpha beta beta"),
        ("a b c d e f g", "g f e d c b a"),
        ("Metformin 二甲双胍 lowers 血糖 glucose", "二甲双胍 lowers glucose levels 降低血糖"),
    ]
    all_agree = True
    detail = ""
    for hyp, ref in pairs:
        lib = ev.compute_rouge(hyp, ref, backend="rouge")
        own = ev.compute_rouge(hyp, ref, backend="builtin")
        for name in ("rouge-1", "rouge-2", "rouge-l"):
            for stat in ("r", "p", "f"):
                if not close(lib[name][stat], own[name][stat], 1e-6):
                    all_agree = False
                    detail = f"({hyp[:20]}... {name} {stat}: 库={lib[name][stat]:.6f} 自写={own[name][stat]:.6f})"
    check("rouge库和自己写的简化版结果一致（6组文本 x 3种ROUGE x r/p/f）", all_agree, detail)
    # 证明为什么23里要用exclusive=False：库默认的exclusive=True会让重复词只算一次，这里直接对比两种结果
    from rouge import Rouge
    default_mode = Rouge().get_scores("a a a", "a")[0]["rouge-1"]["f"]
    ours = ev.compute_rouge("a a a", "a", backend="rouge")["rouge-1"]["f"]
    check("rouge库默认会把重复词只算一次('a a a' 对 'a' 得1.0)，这里用exclusive=False按次数算(0.5)",
          close(default_mode, 1.0) and close(ours, 0.5), f"(库默认={default_mode:.3f}，这里={ours:.3f})")
    auto = ev.compute_rouge("alpha beta", "alpha beta")
    check("backend=auto 装了库就用库", auto["backend"] == "rouge")


def test_key_info():
    """检查关键信息提取（百分比、剂量、频率、时间、安全/治疗/机制词，中英文统一）和召回率的计算（部分覆盖、None、去重等）"""
    print("\n--- 测试2：关键信息提取和召回 ---")
    info = ev.extract_key_info("Reduced risk by 25% and 15-25 percent; 百分之30也有；dose 75-100 mg once daily, 10 mg/kg, 500毫克。")
    check("百分比：25%、15-25%、百分之30", info["percent"] == {"25%", "15-25%", "30%"}, f"(得到 {sorted(info['percent'])})")
    check("剂量：75-100mg、10mg/kg、500mg、每日一次", info["dosage"] == {"75-100mg", "10mg/kg", "500mg", "freq:1/day"},
          f"(得到 {sorted(info['dosage'])})")

    info = ev.extract_key_info("持续12个月，随访3 weeks，24小时内，2-3天，发表于2021年，随访5年")
    check("时间：12个月 3 weeks 24小时 2-3天 5年，不含年份2021年", info["time"] == {"12month", "3week", "24hour", "2-3day", "5year"},
          f"(得到 {sorted(info['time'])})")

    en = ev.extract_key_info("treated for 12 months and 3 weeks, 24 hours")
    zh = ev.extract_key_info("治疗12个月和3周，24小时")
    check("中英文单位统一：months/个月、weeks/周、hours/小时", en["time"] == zh["time"], f"({sorted(en['time'])} vs {sorted(zh['time'])})")
    check("中英文单位统一：mg/毫克", ev.extract_key_info("75 mg")["dosage"] == ev.extract_key_info("75毫克")["dosage"])
    check("频率：twice daily / 每日两次 / bid", ev.extract_key_info("twice daily")["dosage"] == ev.extract_key_info("每日两次")["dosage"]
          == ev.extract_key_info("bid")["dosage"] == {"freq:2/day"})
    check("浓度(mg/dL、mg/L)不是剂量", not ev.extract_key_info("LDL 130 mg/dL, 5 mg/L")["dosage"])
    check("数字后面紧跟英文字母的不是剂量（5 mgx / 3gram不算）", not ev.extract_key_info("5mgx 3gram")["dosage"])

    info = ev.extract_key_info("Side effects and adverse events; risk; contraindication")
    check("安全信息（英文）：side_effect adverse risk contraindication",
          info["safety"] == {"side_effect", "adverse", "risk", "contraindication"}, f"(得到 {sorted(info['safety'])})")
    info = ev.extract_key_info("有副作用、不良反应和风险，禁忌症")
    check("安全信息（中文）和英文得到同样的标准名称",
          info["safety"] == {"side_effect", "adverse", "risk", "contraindication"}, f"(得到 {sorted(info['safety'])})")

    info = ev.extract_key_info("建议采用该治疗方案")
    check("治疗建议（中文）：recommend treatment regimen", info["treatment"] == {"recommend", "treatment", "regimen"})
    info = ev.extract_key_info("We recommend this treatment regimen; therapy is effective")
    check("治疗建议（英文）：recommend treatment regimen", info["treatment"] == {"recommend", "treatment", "regimen"})

    info = ev.extract_key_info("作用机制是抑制血小板")
    check("作用机制（中文）：机制 + 作用", info["mechanism"] == {"mechanism", "action"})
    info = ev.extract_key_info("有一些副作用")
    check("'副作用'里的'作用'不算作用机制", info["mechanism"] == set() and info["safety"] == {"side_effect"})
    check("没有关键信息的文本 -> 全是空", all(not v for v in ev.extract_key_info("今天天气很好").values()))

    # 召回率：reference里9条关键信息
    reference = ("Take 75-100 mg once daily for 12 months; it reduces risk by 25%. "
                 "Recommended treatment regimen. Mechanism: inhibits platelet.")
    full_zh = "每日一次，每次75-100毫克，持续12个月，风险降低25%。建议采用该治疗方案。作用机制是抑制血小板。"
    r = ev.compute_key_info_recall(full_zh, reference)
    check("标准答案提取到9条关键信息", r["overall_gt_matches"] == 9, f"(得到 {r['overall_gt_matches']})")
    check("中文答案覆盖全部 -> 总召回1.0（跨语言）", r["overall_recall"] == 1.0, f"(得到 {r['overall_recall']})")
    check("多出来的'action'记在extra里，不影响召回", r["per_category"]["mechanism"]["extra"] == ["action"])

    r = ev.compute_key_info_recall("每日一次，25%。", reference)
    check("部分覆盖：总召回 2/9", close(r["overall_recall"], 2 / 9), f"(得到 {r['overall_recall']})")
    check("部分覆盖：各类别召回 百分比1、剂量0.5、时间0、安全0、建议0、机制0",
          (r["per_category"]["percent"]["recall"], r["per_category"]["dosage"]["recall"],
           r["per_category"]["time"]["recall"], r["per_category"]["safety"]["recall"],
           r["per_category"]["treatment"]["recall"], r["per_category"]["mechanism"]["recall"]) == (1.0, 0.5, 0.0, 0.0, 0.0, 0.0))
    check("漏掉的项目列出来", r["per_category"]["dosage"]["missed"] == ["75-100mg"] and r["per_category"]["time"]["missed"] == ["12month"])
    r = ev.compute_key_info_recall(reference, reference)
    check("标准答案和自己比 -> 召回1.0", r["overall_recall"] == 1.0)
    r = ev.compute_key_info_recall("什么都没有", "今天天气很好")
    check("标准答案没有关键信息 -> 召回是None而不是除0报错", r["overall_recall"] is None
          and all(c["recall"] is None for c in r["per_category"].values()))
    r = ev.compute_key_info_recall("", reference)
    check("生成答案为空 -> 召回0", r["overall_recall"] == 0.0)
    r = ev.compute_key_info_recall("25% 25% 25%", "25%")
    check("重复出现的只算一次（去重）", r["overall_gt_matches"] == 1 and r["overall_recall"] == 1.0)


def test_hallucination():
    """检查幻觉检测：四种信号各自什么时候触发/不触发、风险分和档位、引用标记计数，以及"数值不在上下文里"这个额外信号"""
    print("\n--- 测试3：幻觉检测 ---")

    def names(text, context=None):
        return [s["name"] for s in ev.detect_hallucination(text, context)["signals"]]

    check("研究表明 + 没有引用 -> 触发", names("研究表明二甲双胍有效。") == ["study_claim_no_citation"])
    check("研究表明 + 同一句有[1]引用 -> 不触发", names("研究表明二甲双胍有效[1]。") == [])
    check("研究表明 + 同一句有[来源: ...] -> 不触发", names("研究表明二甲双胍有效[来源: Some paper]。") == [])
    check("Studies show + 没有引用 -> 触发", names("Studies show that it works.") == ["study_claim_no_citation"])
    check("Studies have shown + 有作者年份引用 -> 不触发", names("Studies have shown benefit (Smith et al., 2020).") == [])
    check("引用在别的句子里不算（要在同一句）", names("研究表明二甲双胍有效。另一句话[1]。") == ["study_claim_no_citation"])

    check("已被证明 + 没有限定 -> 触发", names("该药已被证明有效。") == ["proven_no_qualifier"])
    check("已被证明 + 有限定条件（在老年患者中）-> 不触发", names("该药在老年患者中已被证明有效。") == [])
    check("已经证明 + 有'可能' -> 不触发", names("它可能已经证明有效。") == [])
    check("has been proven + 没限定 -> 触发", names("It has been proven to work.") == ["proven_no_qualifier"])
    check("is proven + 有'in patients with' -> 不触发", names("It is proven in patients with diabetes.") == [])

    check("100% -> 触发", names("治愈率为100%。") == ["hundred_percent"])
    check("百分之百 -> 触发", names("百分之百有效。") == ["hundred_percent"])
    check("100 percent -> 触发", names("It works in 100 percent of cases.") == ["hundred_percent"])
    check("1100% 不是100%", names("增长了1100%。") == [])
    check("50-100% 的范围不触发", names("缓解率为50-100%。") == [])
    check("99.9% 不触发", names("缓解率为99.9%。") == [])

    check("完全安全 -> 触发", names("这个药完全安全。") == ["completely_safe_effective"])
    check("完全有效 / 完全无害 -> 触发", names("完全有效。") == ["completely_safe_effective"] and names("完全无害。") == ["completely_safe_effective"])
    check("completely safe -> 触发", names("The drug is completely safe.") == ["completely_safe_effective"])
    check("'完全'后面不是这几个词 -> 不触发", names("完全不同的两个机制。") == [])

    clean = ev.detect_hallucination("二甲双胍可能与较少的心血管事件相关[1]，证据有限。")
    check("正常谨慎的表述：0个信号，风险低，风险分0", clean["signal_count"] == 0 and clean["risk_level"] == "低" and clean["risk_score"] == 0)
    one = ev.detect_hallucination("该药已被证明有效。")
    two = ev.detect_hallucination("该药已被证明有效。治愈率为100%。")
    four = ev.detect_hallucination("研究表明该药已被证明有效。治愈率为100%，完全安全。")
    check("信号越多风险分越高（1个<2个<4个）", one["risk_score"] < two["risk_score"] < four["risk_score"],
          f"({one['risk_score']:.2f} < {two['risk_score']:.2f} < {four['risk_score']:.2f})")
    check("风险档位：1个=中，2个=高，4个=高且分数封顶1.0",
          one["risk_level"] == "中" and two["risk_level"] == "高" and four["risk_score"] == 1.0 and four["signal_count"] == 4)
    check("引用标记个数统计", ev.detect_hallucination("A[1]。B[2]。C[来源: x]。")["citation_markers"] == 3)

    # 额外信号：数值不在上下文里
    context = "Statins reduced stroke risk by 21% in the trials over 5 years."
    r = ev.detect_hallucination("他汀使卒中风险降低21%。", context)
    check("数值在上下文里 -> 不触发额外信号", r["signal_count"] == 0 and r["context_checked"])
    r = ev.detect_hallucination("他汀使卒中风险降低47%。", context)
    check("数值不在上下文里 -> 触发number_not_in_context", [s["name"] for s in r["signals"]] == ["number_not_in_context"])
    check("额外信号权重0.5", close(r["weighted_count"], 0.5))
    r = ev.detect_hallucination("他汀使卒中风险降低47%。")
    check("没给上下文 -> 不做这项检查", r["signal_count"] == 0 and not r["context_checked"])
    many = "降低11%、22%、33%、44%。"
    r = ev.detect_hallucination(many, context)
    check("额外信号最多算2个", r["signal_count"] == 2)


def test_readability():
    """检查句子切分（缩写、小数点、中文标点、换行）、句长统计，以及可读性分数在正常/超长/太短/空文本时的表现"""
    print("\n--- 测试4：可读性 ---")
    s = ev.split_sentences("Drugs were used, e.g. aspirin. It worked well. HbA1c fell by 1.5 points. Good!")
    check("句子切分：e.g.和小数点不断句", len(s) == 4, f"(得到 {s})")
    s = ev.split_sentences("et al. found that it works. Fig. 2 shows data vs. placebo.")
    check("句子切分：et al. Fig. vs. 不断句", len(s) == 2, f"(得到 {s})")
    s = ev.split_sentences("二甲双胍可以降低血糖。它也可能有副作用！是否安全？需要更多证据；先观察。")
    check("句子切分：中文句末标点", len(s) == 5, f"(得到 {s})")
    s = ev.split_sentences("- 第一条没有句号\n- 第二条也没有\n\n- 第三条")
    check("句子切分：换行分隔的列表项各算一句", len(s) == 3)
    check("句长：英文单词1个算1，汉字1个算1，标点不算", ev.sentence_length("二甲双胍可以降低血糖。") == 10
          and ev.sentence_length("Metformin lowers glucose.") == 3 and ev.sentence_length("Metformin 降低 glucose") == 4)

    ok = ev.compute_readability("二甲双胍可以降低血糖。它也可能引起胃肠道不适。需要在医生指导下使用。")
    check("正常句子：3句，平均句长在10~25之间，分数1.0", ok["sentence_count"] == 3 and 10 <= ok["avg_sentence_length"] <= 25
          and close(ok["readability_score"], 1.0), f"(平均句长{ok['avg_sentence_length']:.1f}，分数{ok['readability_score']:.2f})")
    long_text = " ".join(["word"] * 80) + "."
    lg = ev.compute_readability(long_text)
    check("一句80个词的超长句：长句比例100%，分数0", lg["long_sentence_ratio"] == 1.0 and lg["readability_score"] == 0
          and lg["max_sentence_length"] == 80)
    mixed = ev.compute_readability(long_text + " Short one here today.")
    check("超长句拉低分数：加一个短句后分数仍低于正常文本", mixed["readability_score"] < ok["readability_score"] and mixed["long_sentence_ratio"] == 0.5)
    tiny = ev.compute_readability("好。")
    check("太短的句子分数低于1", tiny["readability_score"] < 1)
    empty = ev.compute_readability("")
    check("空文本：不报错，分数0", empty["sentence_count"] == 0 and empty["readability_score"] == 0)
    para = ev.compute_readability("第一段内容在这里。\n\n第二段内容在这里。")
    check("段落数：2", para["paragraph_count"] == 2)
    check("英文平均词长", close(ev.compute_readability("ab abcd")["avg_english_word_length"], 3.0))


def test_evaluator_class():
    """检查 AnswerEvaluator 整体：去掉参考文献/免责声明、有无标准答案、body_only，以及接上（假Ollama跑出来的）pipeline结果"""
    print("\n--- 测试5：AnswerEvaluator 整体 ---")
    evaluator = AnswerEvaluator(rouge_backend="builtin")
    body = "二甲双胍可能降低风险25%[1]，建议治疗12个月。"
    answer = body + "\n\n参考文献：\n[1] Some paper\n\n免责声明：本回答基于检索到的文献证据整理生成，不构成医疗建议，具体诊疗请遵医嘱。"
    check("get_answer_body：去掉参考文献和免责声明", ev.get_answer_body(answer) == body)
    check("get_answer_body：检索到的相关文献也去掉",
          ev.get_answer_body("正文\n\n检索到的相关文献（回答中没有标注具体引用）：\n[1] x\n\n免责声明：y") == "正文")

    r = evaluator.evaluate(answer, reference="Reduces risk by 25% with 12 months of treatment recommended.")
    check("评估报告包含四部分", all(k in r for k in ("similarity", "key_info", "hallucination", "readability")))
    check("只评估正文（body_chars < 整个答案）", r["body_chars"] == len(body) and r["answer_chars"] > r["body_chars"])
    check("跨语言关键信息命中：25% 12month risk treatment recommend",
          r["key_info"]["overall_recall"] == 1.0, f"(得到 {r['key_info']['overall_recall']})")
    r2 = evaluator.evaluate(answer)
    check("不给标准答案：相似性和关键信息为None，幻觉和可读性仍然有", r2["similarity"] is None and r2["key_info"] is None
          and r2["hallucination"] is not None and r2["readability"]["sentence_count"] > 0)
    r3 = evaluator.evaluate(answer, reference="Reduces risk by 25%.", body_only=False)
    check("body_only=False：评估整个答案", r3["body_chars"] == len(answer))

    self_report = evaluator.evaluate(body, reference=body)
    check("标准答案和自己比：ROUGE-1/2/L的F1都是1", all(close(self_report["similarity"][k]["f"], 1.0) for k in ("rouge-1", "rouge-2", "rouge-l")))
    check("标准答案和自己比：关键信息召回1.0", self_report["key_info"]["overall_recall"] == 1.0)

    # 接pipeline的结果
    pipeline, llm = make_pipeline(FULL_SCRIPT)
    result = pipeline.generate("q", make_docs())
    check("pipeline结果里保存了上下文文本（组装后和筛选后）",
          bool(result["intermediate_results"]["context_text"]) and bool(result["intermediate_results"]["context_text_filtered"])
          and TITLE_1 in result["intermediate_results"]["context_text"])
    check("筛选后的上下文去掉了无关文档", "zebrafish" in result["intermediate_results"]["context_text"].lower()
          and "zebrafish" not in result["intermediate_results"]["context_text_filtered"].lower())
    rep = evaluator.evaluate_result(result, reference="Metformin was associated with fewer cardiovascular events in type 2 diabetes.")
    check("evaluate_result：自动带上上下文做数值核对", rep["hallucination"]["context_checked"])
    check("evaluate_result：给了标准答案就有ROUGE", rep["similarity"] is not None)
    text = ev.format_evaluation_report(rep, "测试")
    check("format_evaluation_report 能输出", "文本相似性" in text and "幻觉风险" in text and "可读性" in text)
    text2 = ev.format_evaluation_report(evaluator.evaluate("x"), None)
    check("format_evaluation_report：没有标准答案也能输出", "跳过" in text2)


# ===========================================================================
# 二、缓存
# ===========================================================================
class FakeClock:
    """假时钟：像time.time一样可以调用，但时间由测试自己拨（改 now），这样测TTL不用真的等"""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def fake_result(answer="答案", ok=True, size_pad=0):
    """造一个最简单的"生成结果"字典。ok是答案生成阶段成功与否，size_pad往答案后面补这么多个x，用来控制大小"""
    return {"query": "q", "answer": answer + "x" * size_pad,
            "generation_metrics": {"stage_success": {"answer_generation": ok}, "total_time_seconds": 5.0}}


def test_cache_key():
    """检查缓存键：相同输入相同键、64位十六进制、设置顺序无关、查询/上下文/模型/温度任何一个变了键就变、空白和大小写的处理"""
    print("\n--- 测试6：缓存键 ---")
    base = make_cache_key("问题", "上下文", {"model": "m", "t": 0.1})
    check("相同输入 -> 相同的键", base == make_cache_key("问题", "上下文", {"model": "m", "t": 0.1}))
    check("键是64位十六进制（SHA-256）", len(base) == 64 and all(c in "0123456789abcdef" for c in base))
    check("设置字典的key顺序不影响", base == make_cache_key("问题", "上下文", {"t": 0.1, "model": "m"}))
    check("查询不同 -> 键不同", base != make_cache_key("问题2", "上下文", {"model": "m", "t": 0.1}))
    check("上下文不同 -> 键不同", base != make_cache_key("问题", "上下文2", {"model": "m", "t": 0.1}))
    check("模型不同 -> 键不同", base != make_cache_key("问题", "上下文", {"model": "m2", "t": 0.1}))
    check("温度不同 -> 键不同", base != make_cache_key("问题", "上下文", {"model": "m", "t": 0.2}))
    check("查询的多余空白不影响", make_cache_key("a  b\n c", "x", {}) == make_cache_key(" a b c ", "x", {}))
    check("查询大小写有区别（不转小写）", make_cache_key("EGFR", "x", {}) != make_cache_key("egfr", "x", {}))
    check("查询和上下文的拼接不会混淆（'ab'+'c' vs 'a'+'bc'）", make_cache_key("ab", "c", {}) != make_cache_key("a", "bc", {}))


def test_cache_basic():
    """检查 GenerationCache 的基本功能：命中/未命中、深拷贝、TTL过期（假时钟）、LRU淘汰、字节上限、重复存同一个键、clear"""
    print("\n--- 测试7：GenerationCache 基本功能 ---")
    clock = FakeClock()
    cache = GenerationCache(max_entries=3, max_bytes=10 ** 6, ttl_seconds=100, time_fn=clock)

    got, info = cache.get("k1")
    check("没存过 -> 未命中", got is None and cache.stats()["misses"] == 1)

    stored, reason = cache.put("k1", fake_result("A"), temperature=0.1)
    got, info = cache.get("k1")
    check("存入后命中，内容一致", stored and got["answer"] == "A" and cache.stats()["hits"] == 1)

    got["answer"] = "被改了"
    got2, _ = cache.get("k1")
    check("取出的是深拷贝：改它不影响缓存里的", got2["answer"] == "A")
    original = fake_result("B")
    cache.put("k2", original, 0.1)
    original["answer"] = "存了之后再改原对象"
    got3, _ = cache.get("k2")
    check("存进去的也是拷贝：改原对象不影响缓存", got3["answer"] == "B")

    # TTL
    clock.now += 99
    got, info = cache.get("k1")
    check("TTL：99秒（TTL=100）还没过期", got is not None and abs(info["age_seconds"] - 99) < 1e-9)
    clock.now += 2
    got, _ = cache.get("k1")
    check("TTL：超过100秒后过期，取不到", got is None and cache.stats()["expirations"] >= 1)
    check("过期的条目被删掉了", "k1" not in cache)

    # LRU：容量3
    cache = GenerationCache(max_entries=3, max_bytes=10 ** 6, ttl_seconds=1000, time_fn=clock)
    for k in ("a", "b", "c"):
        cache.put(k, fake_result(k), 0.1)
    # 先取一次a再存d：如果是"先进先出"，被淘汰的会是a；LRU的话应该是b
    cache.get("a")                                  # 用一下a，现在最久没用的是b
    cache.put("d", fake_result("d"), 0.1)
    check("LRU：超过条数上限时淘汰最久没用的（b），不是最早存的（a）",
          "b" not in cache and "a" in cache and "c" in cache and "d" in cache and len(cache) == 3)
    check("淘汰次数统计", cache.stats()["evictions"] == 1)

    # 字节上限
    cache = GenerationCache(max_entries=100, max_bytes=2500, ttl_seconds=1000, time_fn=clock)
    for k in ("x1", "x2", "x3", "x4"):
        cache.put(k, fake_result(k, size_pad=1000), 0.1)
    stats = cache.stats()
    check("字节上限：总字节数不超过上限", stats["total_bytes"] <= 2500, f"({stats['total_bytes']}字节，{stats['entries']}条)")
    check("字节上限：最近存的还在，最早的被淘汰", "x4" in cache and "x1" not in cache)
    stored, reason = cache.put("huge", fake_result("h", size_pad=5000), 0.1)
    check("单条就超过总上限 -> 不缓存，也不会把别的挤掉", (not stored) and "x4" in cache and cache.stats()["skipped_too_large"] == 1)

    # 重复存同一个键：字节数不重复累计
    cache = GenerationCache(max_entries=10, max_bytes=10 ** 6, ttl_seconds=1000, time_fn=clock)
    cache.put("same", fake_result("v1", size_pad=500), 0.1)
    bytes_one = cache.stats()["total_bytes"]
    cache.put("same", fake_result("v1", size_pad=500), 0.1)
    check("重复存同一个键：不重复累计字节数", cache.stats()["total_bytes"] == bytes_one and len(cache) == 1)

    cache.clear()
    check("clear清空", len(cache) == 0 and cache.stats()["total_bytes"] == 0)


def test_cache_gating():
    """检查什么情况下不缓存：温度太高（刚好等于上限要缓存）、温度未知、答案生成失败/被跳过、答案为空或结构不完整，以及对应的统计计数"""
    print("\n--- 测试8：只缓存低温度、成功的结果 ---")
    cache = GenerationCache(max_entries=10, max_temperature=0.3)
    stored, _ = cache.put("t_low", fake_result(), 0.1)
    stored_edge, _ = cache.put("t_edge", fake_result(), 0.3)
    stored_high, reason = cache.put("t_high", fake_result(), 0.31)
    stored_hot, _ = cache.put("t_hot", fake_result(), 0.8)
    check("温度0.1、0.3（刚好等于上限）-> 缓存", stored and stored_edge)
    check("温度0.31、0.8 -> 不缓存", (not stored_high) and (not stored_hot) and "t_high" not in cache, f"({reason})")
    check("统计：因温度高没缓存2次", cache.stats()["skipped_high_temp"] == 2)
    stored, _ = cache.put("t_none", fake_result(), None)
    check("温度未知(None) -> 不缓存", not stored)

    stored, _ = cache.put("bad1", fake_result(ok=False), 0.1)
    check("答案生成阶段失败 -> 不缓存", not stored and "bad1" not in cache)
    skipped = fake_result()
    skipped["generation_metrics"]["stage_success"]["answer_generation"] = None
    stored, _ = cache.put("bad2", skipped, 0.1)
    check("答案生成阶段被跳过(None，如没有上下文) -> 不缓存", not stored)
    stored, _ = cache.put("bad3", fake_result(answer=""), 0.1)
    stored2, _ = cache.put("bad4", {"answer": "x"}, 0.1)
    check("答案为空 / 结构不完整 -> 不缓存", (not stored) and (not stored2))
    check("统计：因结果失败没缓存4次", cache.stats()["skipped_bad_result"] == 4)


def test_cache_threads():
    """检查多个线程同时存取缓存：不报错、条数不超上限、记的总字节数和实际各条大小之和一致、命中+未命中=查询总次数"""
    print("\n--- 测试9：缓存多线程 ---")
    cache = GenerationCache(max_entries=10, max_bytes=10 ** 6, ttl_seconds=1000)
    errors = []

    def worker(n):
        try:
            for i in range(200):
                key = f"k{(n * 7 + i) % 25}"
                cache.put(key, fake_result(key, size_pad=50), 0.1)
                cache.get(key)
                cache.get(f"k{i % 25}")
        except Exception as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = cache.stats()
    # 直接读缓存内部的_data，自己重新加一遍，和cache记的_total_bytes对比
    real_total = sum(e["size"] for e in cache._data.values())
    check("8个线程同时存取：没有异常", not errors, str(errors[:2]))
    check("条数没超过上限", stats["entries"] <= 10, f"({stats['entries']})")
    check("记录的总字节数 = 各条目大小之和（没有记账错误）", stats["total_bytes"] == real_total)
    check("命中+未命中 = 查询总次数", stats["hits"] + stats["misses"] == 8 * 200 * 2)


# ---------------------------------------------------------------------------
# CachedPipeline
# ---------------------------------------------------------------------------
class SlowScriptedLLM(ScriptedLLM):
    # 每次请求先睡一会，模拟真实模型的耗时（用来测并发合并、耗时记录、加速）
    def __init__(self, script, delay=0.1):
        super().__init__(script)
        self.delay = delay

    def _request(self, method, path, payload=None, timeout=None):
        time.sleep(self.delay)
        return super()._request(method, path, payload, timeout)


class PickyLLM(SlowScriptedLLM):
    # 回答阶段遇到问题里有 FAILME 就报错，用来制造"某一个任务的模型调用失败"
    def _request(self, method, path, payload=None, timeout=None):
        system_text = payload["messages"][0]["content"]
        if "医学文献问答助手" in system_text and "FAILME" in payload["messages"][-1]["content"]:
            self.calls.append(("answer_generator", payload))
            raise TimeoutError("模拟请求超时")
        return super()._request(method, path, payload, timeout)


def count_calls(llm, stage):
    """假LLM收到过多少次某个阶段（如"answer_generator"）的请求"""
    return sum(1 for name, _ in llm.calls if name == stage)


def make_cached(script=None, stages=None, cache=None, llm_class=ScriptedLLM, **llm_kwargs):
    """
    用假Ollama搭一个 CachedPipeline。script不传就用22里的FULL_SCRIPT，stages是各阶段提示词/温度（不传用默认的深拷贝），
    cache不传就新建；llm_class可以换成会睡觉的版本。返回 (CachedPipeline, 假llm)
    """
    _m_context_assembler = importlib.import_module("14_context_assembler")
    ContextAssembler = _m_context_assembler.ContextAssembler
    _m_medical_generation_pipeline = importlib.import_module("20_medical_generation_pipeline")
    MedicalGenerationPipeline = _m_medical_generation_pipeline.MedicalGenerationPipeline
    _m_verify_generation_logic = importlib.import_module("22_verify_generation_logic")
    FakeTokenizer = _m_verify_generation_logic.FakeTokenizer
    llm = llm_class(script or FULL_SCRIPT, **llm_kwargs)
    pipeline = MedicalGenerationPipeline(
        assembler=ContextAssembler(tokenizer=FakeTokenizer(), max_context_tokens=2000), llm=llm,
        # 深拷贝：测试里会改某个阶段的温度、提示词，不能改到全局的PROMPT_STAGES
        prompt_stages=stages if stages is not None else copy.deepcopy(PROMPT_STAGES))
    return CachedPipeline(pipeline, cache=cache), llm


def test_cached_pipeline():
    """检查 CachedPipeline 的命中逻辑：第一次生成并存入、第二次命中不再调模型、命中时的耗时记录、结果是拷贝、问题多空白仍命中，
    以及换问题/换上下文/换流程设置/use_cache=False 都不命中"""
    print("\n--- 测试10：CachedPipeline（假Ollama）---")
    cp, llm = make_cached()
    r1 = cp.generate("二甲双胍对心血管有什么影响？", make_docs())
    check("第一次：未命中，调用了4次模型（评估/草稿/审查/最终）", r1["cache"]["hit"] is False and len(llm.calls) == 4)
    check("第一次：已存入缓存（默认温度都<=0.3）", r1["cache"]["stored"] is True and len(cp.cache) == 1,
          f"(最高温度{r1['cache'].get('temperature')})")

    r2 = cp.generate("二甲双胍对心血管有什么影响？", make_docs())
    check("第二次：命中，没有新的模型调用", r2["cache"]["hit"] is True and len(llm.calls) == 4)
    check("命中的答案和第一次完全一样", r2["answer"] == r1["answer"] and r2["sources"] == r1["sources"])
    check("命中：记录了原来生成花的时间", r2["cache"]["original_time_seconds"] == r1["generation_metrics"]["total_time_seconds"])
    # 用会睡觉的假LLM才能看出"命中时耗时约0"和"原来的耗时"的区别（4次请求x0.05秒）
    cps, _ = make_cached(llm_class=SlowScriptedLLM, delay=0.05)
    slow1 = cps.generate("q", make_docs())
    slow2 = cps.generate("q", make_docs())
    check("命中：total_time_seconds是这次实际花的时间（约0），原来的耗时记在cache里",
          slow1["generation_metrics"]["total_time_seconds"] >= 0.2 and slow2["generation_metrics"]["total_time_seconds"] < 0.1
          and slow2["cache"]["original_time_seconds"] >= 0.2,
          f"(第一次{slow1['generation_metrics']['total_time_seconds']:.2f}秒，命中{slow2['generation_metrics']['total_time_seconds']:.3f}秒)")
    r2["answer"] = "被改了"
    r3 = cp.generate("二甲双胍对心血管有什么影响？", make_docs())
    check("改命中返回的结果，不影响缓存里的", r3["answer"] == r1["answer"])

    r4 = cp.generate("  二甲双胍对心血管有什么影响？\n", make_docs())
    check("问题前后多了空白：仍然命中", r4["cache"]["hit"] is True)

    n = len(llm.calls)
    r5 = cp.generate("另一个问题", make_docs())
    check("换了问题 -> 未命中，重新生成", r5["cache"]["hit"] is False and len(llm.calls) == n + 4)

    n = len(llm.calls)
    changed_docs = make_docs()
    changed_docs[0]["text"] = "完全不同的检索内容 Metformin was linked to fewer events in a new cohort."
    r6 = cp.generate("二甲双胍对心血管有什么影响？", changed_docs)
    check("同一问题但检索到的上下文变了 -> 未命中", r6["cache"]["hit"] is False and len(llm.calls) == n + 4)

    n = len(llm.calls)
    r7 = cp.generate("二甲双胍对心血管有什么影响？", make_docs(), enable_review=False)
    check("关闭审查（流程设置不同）-> 键不同，未命中", r7["cache"]["hit"] is False and len(llm.calls) == n + 2)

    n = len(llm.calls)
    r8 = cp.generate("二甲双胍对心血管有什么影响？", make_docs(), use_cache=False)
    check("use_cache=False：绕过缓存，重新生成", r8["cache"]["hit"] is False and len(llm.calls) == n + 4)

    stats = cp.stats()
    check("统计：命中3次（第二、三、四次）", stats["hits"] == 3, f"({format_cache_stats(stats)})")


def test_cached_pipeline_gating():
    """检查缓存和流程设置的关系：高温不缓存（没用到的高温阶段不影响）、草稿失败不缓存、没有上下文不调模型，
    以及共用同一个缓存时，模型名/提示词模板/温度变了不会误命中，TTL过期后重新生成"""
    print("\n--- 测试11：温度/失败/模板变化与缓存 ---")
    hot_stages = copy.deepcopy(PROMPT_STAGES)
    hot_stages["answer_generator"].temperature = 0.8
    cp, llm = make_cached(stages=hot_stages)
    r1 = cp.generate("q", make_docs())
    r2 = cp.generate("q", make_docs())
    check("答案生成器温度0.8：不缓存", r1["cache"]["stored"] is False and len(cp.cache) == 0)
    check("温度高：第二次也要重新生成", r2["cache"]["hit"] is False and len(llm.calls) == 8)
    check("温度高：统计里记了一次跳过", cp.stats()["skipped_high_temp"] == 2)

    # 温度高的阶段没有被用到时，不影响缓存：关掉审查后，reviewer(0.9)/final不参与
    stages = copy.deepcopy(PROMPT_STAGES)
    stages["critical_reviewer"].temperature = 0.9
    cp, llm = make_cached(stages=stages)
    ra = cp.generate("q", make_docs())
    rb = cp.generate("q", make_docs(), enable_review=False)
    check("审查器温度0.9：开审查时不缓存", ra["cache"]["stored"] is False)
    check("审查器温度0.9：关掉审查后实际没用到它，可以缓存", rb["cache"]["stored"] is True)

    # 失败不缓存
    cp, llm = make_cached(script=dict(FULL_SCRIPT, answer_generator="ERROR"))
    r1 = cp.generate("q", make_docs())
    r2 = cp.generate("q", make_docs())
    check("草稿生成失败：不缓存，下次重新尝试", r1["cache"]["stored"] is False and r2["cache"]["hit"] is False and len(cp.cache) == 0)
    check("草稿生成失败：结果结构仍然完整", r1["answer"].startswith("答案生成失败"))

    # 没有上下文
    cp, llm = make_cached()
    r = cp.generate("q", [])
    check("没有检索结果：不调用模型，不缓存", len(llm.calls) == 0 and r["cache"]["stored"] is False and len(cp.cache) == 0)

    # 共用同一个缓存：换模型、换模板 -> 不应该命中
    shared = GenerationCache()
    cpa, llma = make_cached(cache=shared)
    cpa.generate("q", make_docs())
    cpb, llmb = make_cached(cache=shared)
    rb = cpb.generate("q", make_docs())
    check("共用缓存、设置完全相同（另一个pipeline对象）-> 命中", rb["cache"]["hit"] is True and len(llmb.calls) == 0)

    cpc, llmc = make_cached(cache=shared)
    llmc.model_name = "another-model:7b"
    rc = cpc.generate("q", make_docs())
    check("换了模型名 -> 不命中", rc["cache"]["hit"] is False and len(llmc.calls) == 4)

    new_stages = copy.deepcopy(PROMPT_STAGES)
    new_stages["answer_generator"].system_prompt += "\n新增一条规则。"
    cpd, llmd = make_cached(cache=shared, stages=new_stages)
    rd = cpd.generate("q", make_docs())
    check("改了提示词模板 -> 不命中", rd["cache"]["hit"] is False and len(llmd.calls) == 4)

    temp_stages = copy.deepcopy(PROMPT_STAGES)
    temp_stages["answer_generator"].temperature = 0.2
    cpe, llme = make_cached(cache=shared, stages=temp_stages)
    re_ = cpe.generate("q", make_docs())
    check("改了温度(0.3->0.2，仍然低温) -> 不命中", re_["cache"]["hit"] is False and len(llme.calls) == 4)

    # TTL 过期后重新生成
    clock = FakeClock()
    cp, llm = make_cached(cache=GenerationCache(ttl_seconds=60, time_fn=clock))
    cp.generate("q", make_docs())
    clock.now += 30
    r_in = cp.generate("q", make_docs())
    clock.now += 40
    r_out = cp.generate("q", make_docs())
    check("TTL：30秒时命中，再过40秒(共70秒>60)过期后重新生成", r_in["cache"]["hit"] and not r_out["cache"]["hit"] and len(llm.calls) == 8)


def test_single_flight():
    """检查并发的相同请求只生成一次（其余线程等着用缓存）、锁表用完会清理，以及不同问题不会互相等待（每个键一把锁）"""
    print("\n--- 测试12：并发的相同请求只生成一次 ---")
    cp, llm = make_cached(llm_class=SlowScriptedLLM, delay=0.1)
    answers = []
    errors = []

    def run():
        try:
            answers.append(cp.generate("同一个问题", make_docs()))
        except Exception as e:
            errors.append(repr(e))

    threads = [threading.Thread(target=run) for _ in range(4)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    check("4个线程同时问同一个问题：没有异常", not errors and len(answers) == 4, str(errors[:2]))
    check("答案生成器只调用了1次（其余线程等待后用缓存）", count_calls(llm, "answer_generator") == 1, f"(调用{count_calls(llm, 'answer_generator')}次)")
    check("4个答案一致", len(set(a["answer"] for a in answers)) == 1)
    check("1个未命中 + 3个命中", sum(1 for a in answers if a["cache"]["hit"]) == 3)
    check("统计里记录了并发合并", cp.stats()["coalesced"] >= 1, f"(coalesced={cp.stats()['coalesced']}，耗时{wall:.2f}秒)")
    check("锁表已清理(没有残留)", len(cp._inflight) == 0)

    # 不同问题可以同时生成（不会互相等待）
    cp, llm = make_cached(llm_class=SlowScriptedLLM, delay=0.1)
    start = time.time()
    threads = [threading.Thread(target=lambda q=q: cp.generate(q, make_docs())) for q in ("问题A", "问题B")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    check("两个不同的问题并行：耗时明显小于串行(8次x0.1秒=0.8秒)", wall < 0.7 and count_calls(llm, "answer_generator") == 2, f"(耗时{wall:.2f}秒)")


# ===========================================================================
# 三、批量处理
# ===========================================================================
class FakeGenerator:
    # 不走pipeline、只有generate()的最简单生成器，用来单独测batch_generate。
    # 假的生成器：每个问题睡指定时间；问题是 'boom' 就抛异常；记录完成顺序和同时在跑的最大线程数
    def __init__(self, delays=None):
        self.delays = delays or {}
        self.finish_order = []
        self.running = 0
        self.max_running = 0
        self.lock = threading.Lock()

    def generate(self, query, docs, **kwargs):
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            time.sleep(self.delays.get(query, 0.01))
            if query == "boom":
                raise RuntimeError("故意失败")
            return {"answer": f"答案:{query}", "kwargs": kwargs,
                    "generation_metrics": {"stage_success": {"answer_generation": True}}}
        finally:
            with self.lock:
                self.running -= 1
                self.finish_order.append(query)


def test_batch_workers():
    """检查 choose_worker_count：默认上限2、不超过CPU核心数和任务数、至少为1、default_cap可以调（cpu_count是手动传的假值，和本机无关）"""
    print("\n--- 测试13：线程数选择 ---")
    check("没指定：min(CPU, 2, 任务数)：8核 -> 2", choose_worker_count(10, None, cpu_count=8) == 2)
    check("没指定：单核 -> 1", choose_worker_count(10, None, cpu_count=1) == 1)
    check("指定8、CPU只有4 -> 4（不超过CPU核心数）", choose_worker_count(10, 8, cpu_count=4) == 4)
    check("指定3、CPU有8 -> 3", choose_worker_count(10, 3, cpu_count=8) == 3)
    check("任务只有1个 -> 1个线程", choose_worker_count(1, 8, cpu_count=8) == 1)
    check("指定0或负数 -> 至少1", choose_worker_count(5, 0, cpu_count=4) == 1 and choose_worker_count(5, -3, cpu_count=4) == 1)
    check("可以调高默认上限：default_cap=4", choose_worker_count(10, None, cpu_count=8, default_cap=4) == 4)


def test_batch_order_and_isolation():
    """检查 batch_generate：输出顺序和输入一致（即使完成顺序不同）、各种坏任务只影响自己、公共参数和options的合并、空列表、进度回调"""
    print("\n--- 测试14：批量处理 顺序和失败隔离 ---")
    docs = make_docs()
    # 第一个任务最慢，会最后完成：这样"完成顺序"和"输入顺序"一定不同，才能证明结果是按下标放回去的，不是谁先完成谁在前
    # （t0要睡0.3秒，其余只睡0.02秒，用2个线程）
    gen = FakeGenerator(delays={"t0": 0.3, "t1": 0.02, "t2": 0.02, "t3": 0.02})
    out = batch_generate(gen, [(f"t{i}", docs) for i in range(4)], max_workers=2)
    records = out["records"]
    check("输出顺序和输入一致", [r["query"] for r in records] == ["t0", "t1", "t2", "t3"] and [r["index"] for r in records] == [0, 1, 2, 3])
    check("每条结果和自己的问题对应", all(r["result"]["answer"] == f"答案:t{i}" for i, r in enumerate(records)))
    check("实际完成顺序和输入顺序不同（t0最后完成）—— 说明确实并行了", gen.finish_order[-1] == "t0" and gen.finish_order != ["t0", "t1", "t2", "t3"],
          f"(完成顺序 {gen.finish_order})")
    check("同时运行的线程数不超过线程数设置", gen.max_running <= out["summary"]["workers"])

    # 失败隔离：混进各种坏任务（None、缺docs、generate抛异常、query是空白、不是任务的字符串），好任务应该不受影响
    gen = FakeGenerator()
    tasks = [("good1", docs), None, {"query": "no-docs"}, ("boom", docs), {"query": "   ", "docs": []}, "字符串不是任务",
             {"query": "good2", "docs": docs, "options": {"enable_review": False}}, ("good3", docs)]
    out = batch_generate(gen, tasks, max_workers=2, generate_kwargs={"enable_evaluation": False})
    ok = [r["success"] for r in out["records"]]
    check("失败隔离：坏任务失败，其余照常成功", ok == [True, False, False, False, False, False, True, True], f"({ok})")
    check("输出条数和输入一致、顺序对应", len(out["records"]) == len(tasks) and [r["index"] for r in out["records"]] == list(range(len(tasks))))
    check("失败的任务有错误说明", all(out["records"][i]["error"] for i in (1, 2, 3, 4, 5)))
    check("generate抛的异常被记录（RuntimeError: 故意失败）", "RuntimeError" in out["records"][3]["error"] and "故意失败" in out["records"][3]["error"])
    check("汇总：成功3失败5", out["summary"]["succeeded"] == 3 and out["summary"]["failed"] == 5)
    check("公共参数和任务自己的options都传下去了（options覆盖公共参数）",
          out["records"][0]["result"]["kwargs"] == {"enable_evaluation": False}
          and out["records"][6]["result"]["kwargs"] == {"enable_evaluation": False, "enable_review": False})

    out = batch_generate(FakeGenerator(), [])
    check("空任务列表：不报错，返回空", out["records"] == [] and out["summary"]["total"] == 0)

    calls = []
    out = batch_generate(FakeGenerator(), [(f"p{i}", docs) for i in range(5)], max_workers=2,
                         progress=lambda done, total, rec: calls.append((done, total, rec["query"])))
    check("进度回调：每个任务一次，done递增", [c[0] for c in calls] == [1, 2, 3, 4, 5] and all(c[1] == 5 for c in calls))
    check("汇总格式化能输出", "共5个" in format_batch_summary(out["summary"]))


def test_batch_with_pipeline():
    """检查批量处理接上真实的pipeline/CachedPipeline代码（假Ollama）：模型调用失败的隔离、批量里重复问题只生成一次、
    上下文组装加锁（不会同时进入两个线程）、2个线程比1个线程快、两种线程数答案一致"""
    print("\n--- 测试15：批量处理 + 真实pipeline代码（假Ollama）---")
    docs = make_docs()
    # 模型调用失败(只影响带FAILME的那个问题)
    cp, llm = make_cached(llm_class=PickyLLM, delay=0.01)
    queries = ["问题一", "FAILME 问题二", "问题三"]
    out = batch_generate(cp, [(q, docs) for q in queries], max_workers=2)
    ok = [r["success"] for r in out["records"]]
    check("某个问题的模型调用失败：只有它失败，其余成功", ok == [True, False, True], f"({ok})")
    check("失败的任务保留了pipeline返回的结果和原因", out["records"][1]["result"] is not None
          and "答案生成阶段失败" in out["records"][1]["error"])
    check("成功的任务答案正常", all(out["records"][i]["result"]["answer"].count("参考文献") >= 1 for i in (0, 2)))
    check("失败的问题没有被缓存，成功的被缓存", len(cp.cache) == 2)

    # 批量 + 缓存：重复问题只生成一次
    cp, llm = make_cached(llm_class=SlowScriptedLLM, delay=0.05)
    qs = ["Q1", "Q2", "Q1", "Q1", "Q2"]
    out = batch_generate(cp, [(q, docs) for q in qs], max_workers=2)
    check("批量里有重复问题：只生成了2次（答案生成器调用2次）", count_calls(llm, "answer_generator") == 2,
          f"(调用{count_calls(llm, 'answer_generator')}次)")
    check("其余3个来自缓存", out["summary"]["cache_hits"] == 3)
    check("相同问题的答案相同，顺序对应", out["records"][0]["result"]["answer"] == out["records"][2]["result"]["answer"]
          == out["records"][3]["result"]["answer"] and [r["query"] for r in out["records"]] == qs)

    # 组装上下文这一步必须加锁（tokenizer不一定线程安全）：给assemble加一个"同时进来两个线程就记录下来"的检查
    for label, use_cached in (("CachedPipeline", True), ("MedicalGenerationPipeline", False)):
        cp, llm = make_cached(llm_class=SlowScriptedLLM, delay=0.01)
        gen = cp if use_cached else cp.pipeline
        state = {"now": 0, "max": 0}
        guard = threading.Lock()
        real_assemble = gen.assembler.assemble

        def watched_assemble(*args, **kwargs):
            with guard:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            try:
                return real_assemble(*args, **kwargs)
            finally:
                with guard:
                    state["now"] -= 1

        gen.assembler.assemble = watched_assemble
        batch_generate(gen, [(f"并发组装{i}", docs) for i in range(4)], max_workers=4)
        check(f"{label}：多线程时上下文组装不会同时进入两个线程", state["max"] == 1, f"(同时最多{state['max']}个)")

    # 速度：线程数>=2时，应该比单线程快（假模型每次请求睡0.1秒，所以主要看并行是不是真的生效）
    # 这里只说明"线程池代码本身是并行的"，不说明真实Ollama会不会更快（见27的C阶段）
    docs_list = [(f"不同的问题{i}", docs) for i in range(4)]
    cp1, llm1 = make_cached(llm_class=SlowScriptedLLM, delay=0.1)
    one = batch_generate(cp1, docs_list, max_workers=1)
    cp2, llm2 = make_cached(llm_class=SlowScriptedLLM, delay=0.1)
    two = batch_generate(cp2, docs_list, max_workers=2)
    if two["summary"]["workers"] >= 2:
        check("2个线程比1个线程快（假模型，4个任务x4次请求x0.1秒）", two["summary"]["wall_seconds"] < one["summary"]["wall_seconds"] * 0.8,
              f"(1线程{one['summary']['wall_seconds']:.2f}秒，2线程{two['summary']['wall_seconds']:.2f}秒)")
    else:
        print("  [SKIP] 这台机器只有1个CPU核心，线程数被限制成1，跳过加速检查")
    check("两种线程数的答案完全一样", [r["result"]["answer"] for r in one["records"]] == [r["result"]["answer"] for r in two["records"]])


# 按"评估器 -> 缓存 -> 批量"的顺序依次跑所有测试，最后打印汇总
if __name__ == "__main__":
    test_rouge()
    test_key_info()
    test_hallucination()
    test_readability()
    test_evaluator_class()
    test_cache_key()
    test_cache_basic()
    test_cache_gating()
    test_cache_threads()
    test_cached_pipeline()
    test_cached_pipeline_gating()
    test_single_flight()
    test_batch_workers()
    test_batch_order_and_isolation()
    test_batch_with_pipeline()

    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s in results if s == "PASS")
    n_fail = sum(1 for _, s in results if s == "FAIL")
    print(f"汇总：PASS={n_pass}  FAIL={n_fail}")
    if n_fail:
        print("失败的检查：")
        for name, status in results:
            if status == "FAIL":
                print("  -", name)
