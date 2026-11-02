"""
生成答案评估 - AnswerEvaluator

从四个角度给一个生成的答案打分（每个角度单独报告，不合成一个总分，因为总分的权重只能拍脑袋定）：

    1. 文本相似性   ROUGE-1/2/L，和标准答案（参考答案）比。优先用rouge库，没装的话退回到自己写的简化版
    2. 关键信息召回  用正则从标准答案和生成答案里提取医学关键信息（百分比、剂量、时间范围、安全信息、
                    治疗建议、作用机制），recall = overlap / gt_matches，即标准答案里的关键信息被生成答案覆盖了多少
    3. 幻觉检测     检查"没有依据的绝对化表述"，信号越多风险越高
    4. 可读性       平均句长、长句比例等

重要的限制（请一起看结果）：
    - ROUGE和关键信息召回只是"和标准答案像不像"，不等于"对不对"。答案写得对但换了说法，分数也会低；
      模型老实回答"证据不足"，也会因为和标准答案差得远而被扣分。
    - 幻觉检测只是看几种常见的说话方式，是启发式的风险信号，不是判决。没触发信号不代表没有幻觉。
    - 中英文混合：英文按单词、中文按单个汉字切分；关键信息的正则同时支持中英文，
      并把"毫克/mg""个月/months"之类换成统一写法，这样中文答案和英文标准答案也能比。

用法（作为模块被别的脚本导入，比如 26_ 和 27_ 就是这样用的）：
    evaluator = AnswerEvaluator()
    report = evaluator.evaluate(answer, reference=标准答案, context_text=上下文)
    report = evaluator.evaluate_result(pipeline生成的result, reference=标准答案)
    print(format_evaluation_report(report))

输入：answer（字符串）；reference（标准答案，可以不给）；context_text（给模型看的上下文，可以不给）。
输出：report字典，有 answer_chars、body_chars、similarity（ROUGE，没有标准答案时是None）、
      key_info（关键信息召回，没有标准答案时是None）、hallucination、readability 这几项。
      format_evaluation_report(report) 可以把它整理成文字。

直接运行：python 23_answer_evaluator.py
    用文件最后写的两段短文本（一个较好的答案、一个夸大的答案）演示一遍，不需要Ollama，没有命令行参数。
"""

import re

# 优先用 rouge 库（pip install rouge）
try:
    from rouge import Rouge as _RougeLib
    HAS_ROUGE_LIB = True
except ImportError:
    _RougeLib = None
    HAS_ROUGE_LIB = False

# 答案末尾的"参考文献/免责声明"部分不是模型写的正文，评估时要去掉（和21_test_generation_pipeline.py的切法一样）
TAIL_SPLIT_PATTERN = re.compile(r"\n\n(?:参考文献|检索到的相关文献|免责声明)")
# 引用标记：后处理之后是[1]，没经过后处理是[来源: xxx]
CITATION_MARK_PATTERN = re.compile(r"\[\d+\]|\[来源[:：][^\]]*\]")

CJK = "一-鿿"
# 英文单词（可以带连字符和撇号）或者单个汉字，用来切分和统计"长度"
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[" + CJK + "]")


# ---------------------------------------------------------------------------
# 文本预处理
# ---------------------------------------------------------------------------
def get_answer_body(answer):
    """去掉参考文献和免责声明，只留模型写的正文"""
    return TAIL_SPLIT_PATTERN.split(answer)[0].strip()


def strip_citations(text):
    """去掉[1]、[来源: xxx]这类引用标记，免得它们算进ROUGE的词里"""
    return CITATION_MARK_PATTERN.sub("", text)


def tokenize_mixed(text):
    """英文按单词（小写）、中文按单个汉字切分，返回词列表"""
    return [w.lower() for w in WORD_PATTERN.findall(text)]


# ---------------------------------------------------------------------------
# 1. 文本相似性 ROUGE
# ---------------------------------------------------------------------------
def _ngram_counts(tokens, n):
    # 统计每个n元组出现了几次（重复的按次数算，和下面exclusive=False的意思一致）
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _prf(overlap, hyp_total, ref_total):
    # 精确率、召回率、F1。和rouge库一样，F1的分母加了一个很小的数避免除0
    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"r": recall, "p": precision, "f": f1}


def _lcs_length(a, b):
    # 最长公共子序列长度（动态规划，只留一行）
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b):
            if x == y:
                current.append(previous[j] + 1)
            else:
                current.append(max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


def builtin_rouge(hyp_tokens, ref_tokens):
    """自己写的简化版ROUGE-1/2/L，输入是词列表。没有装rouge库时用它，也用来和rouge库的结果互相核对"""
    scores = {}
    for n, name in ((1, "rouge-1"), (2, "rouge-2")):
        hyp_counts = _ngram_counts(hyp_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        overlap = 0
        for gram, count in hyp_counts.items():
            overlap += min(count, ref_counts.get(gram, 0))
        scores[name] = _prf(overlap, max(len(hyp_tokens) - n + 1, 0), max(len(ref_tokens) - n + 1, 0))
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    scores["rouge-l"] = _prf(lcs, len(hyp_tokens), len(ref_tokens))
    return scores


# 两段文本的"中文字符占比"相差达到这个值（>=），认为语言不一致（比如中文答案对英文标准答案）
LANGUAGE_MISMATCH_GAP = 0.6
CJK_CHAR_PATTERN = re.compile("[" + CJK + "]")


def cjk_ratio(tokens):
    """词列表里中文字符所占的比例（英文单词算1个，汉字算1个）"""
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if CJK_CHAR_PATTERN.fullmatch(t)) / len(tokens)


def compute_rouge(hypothesis, reference, backend="auto"):
    """
    计算ROUGE-1/2/L。返回 {"backend": ..., "rouge-1": {"r","p","f"}, "rouge-2": ..., "rouge-l": ...}

    先把两段文本切成"英文单词+单个汉字"再用空格连起来，这样中文也能算（rouge库默认只会按空格分词）。
    任何一边为空：rouge库会报错，所以这里直接返回全0。

    语言不一致的提醒：ROUGE只看词/字是否重合。中文答案对英文标准答案（或反过来），
    几乎没有可以重合的词，分数会接近0，但这不代表答案不好。所以结果里带上两边的中文字符占比，
    相差太大时 language_mismatch=True，看分数时要打折扣。
    """
    hyp_tokens = tokenize_mixed(strip_citations(hypothesis))
    ref_tokens = tokenize_mixed(strip_citations(reference))
    if backend == "auto":
        backend = "rouge" if HAS_ROUGE_LIB else "builtin"
    if backend == "rouge" and not HAS_ROUGE_LIB:
        raise ImportError("没有安装rouge库，请先 pip install rouge")

    hyp_cjk = cjk_ratio(hyp_tokens)
    ref_cjk = cjk_ratio(ref_tokens)
    language_info = {"hyp_cjk_ratio": hyp_cjk, "ref_cjk_ratio": ref_cjk,
                     "language_mismatch": abs(hyp_cjk - ref_cjk) >= LANGUAGE_MISMATCH_GAP}

    if not hyp_tokens or not ref_tokens:
        zero = {"r": 0.0, "p": 0.0, "f": 0.0}
        return dict({"backend": backend, "rouge-1": dict(zero), "rouge-2": dict(zero), "rouge-l": dict(zero),
                     "hyp_len": len(hyp_tokens), "ref_len": len(ref_tokens)}, **language_info)

    if backend == "rouge":
        # exclusive=False：重复出现的词按次数算。rouge库默认exclusive=True，会把重复的词只算一次
        # （例如 "a a a" 对 "a" 会得到1.0），而中文按字切分时重复的字很多，所以这里要关掉
        raw = _RougeLib(exclusive=False).get_scores(" ".join(hyp_tokens), " ".join(ref_tokens))[0]
        scores = {name: {k: float(raw[name][k]) for k in ("r", "p", "f")} for name in ("rouge-1", "rouge-2", "rouge-l")}
    else:
        scores = builtin_rouge(hyp_tokens, ref_tokens)
    scores["backend"] = backend
    scores["hyp_len"] = len(hyp_tokens)
    scores["ref_len"] = len(ref_tokens)
    scores.update(language_info)
    return scores


# ---------------------------------------------------------------------------
# 2. 关键信息提取 + 召回率
# ---------------------------------------------------------------------------
NUM = r"\d+(?:\.\d+)?"
RANGE_SEP = r"\s*(?:-|–|—|~|～|to|至|到)\s*"

# 单位统一写法：中文和英文的写成同一个，这样中英文答案能比
UNIT_MAP = {
    "mg": "mg", "毫克": "mg", "g": "g", "克": "g", "mcg": "mcg", "μg": "mcg", "µg": "mcg", "ug": "mcg", "微克": "mcg",
    "ml": "ml", "毫升": "ml", "iu": "iu", "单位": "iu", "u": "iu",
    "mg/kg": "mg/kg", "毫克/公斤": "mg/kg", "毫克/千克": "mg/kg",
    "year": "year", "years": "year", "年": "year",
    "month": "month", "months": "month", "个月": "month", "月": "month",
    "week": "week", "weeks": "week", "周": "week", "星期": "week", "个星期": "week",
    "day": "day", "days": "day", "天": "day", "日": "day",
    "hour": "hour", "hours": "hour", "h": "hour", "小时": "hour", "个小时": "hour",
    "minute": "minute", "minutes": "minute", "min": "minute", "分钟": "minute",
}

# 剂量：数字(可以是范围) + 剂量单位，后面可以带 /kg 之类
DOSE_PATTERN = re.compile(
    r"(?<![0-9A-Za-z.])(" + NUM + r")(?:" + RANGE_SEP + "(" + NUM + r"))?\s*"
    r"(mg/kg|毫克/公斤|毫克/千克|mg|mcg|μg|µg|ug|ml|iu|毫克|微克|毫升|单位|克|g)(?![A-Za-z])(?!\s*/\s*d?l\b)",   # 后面跟 /dL、/L 的是浓度(mg/dL)，不是剂量
    re.IGNORECASE)
# 用药频率
FREQ_PATTERNS = [
    (re.compile(r"(?:once|1 time|one time)\s+(?:a |per |every )?(?:day|daily)|once daily|每[日天]一次|每[日天]1次|\bqd\b", re.I),
     "freq:1/day"),
    (re.compile(r"(?:twice|2 times|two times)\s+(?:a |per |every )?(?:day|daily)|twice daily|每[日天]两次|每[日天]2次|\bbid\b", re.I),
     "freq:2/day"),
    (re.compile(r"(?:three times|3 times)\s+(?:a |per |every )?(?:day|daily)|每[日天]三次|每[日天]3次|\btid\b", re.I),
     "freq:3/day"),
    (re.compile(r"(?:four times|4 times|up to 4 times)\s+(?:a |per |every )?(?:day|daily)|每[日天]四次|每[日天]4次|\bqid\b", re.I),
     "freq:4/day"),
]
# 百分比：25%、15-25%、百分之30（"百分之"后面只认阿拉伯数字，"百分之二十"这种汉字数字认不出来）
PERCENT_PATTERN = re.compile(
    r"(?<![0-9A-Za-z.])(" + NUM + r")(?:" + RANGE_SEP + "(" + NUM + r"))?\s*(?:%|％|percent|per cent)", re.IGNORECASE)
PERCENT_CN_PATTERN = re.compile(r"百分之\s*(" + NUM + r")")
# 时间范围：12个月、3 weeks、24小时、2-3天
TIME_PATTERN = re.compile(
    r"(?<![0-9A-Za-z.])(" + NUM + r")(?:" + RANGE_SEP + "(" + NUM + r"))?\s*"
    r"(years?|months?|weeks?|days?|hours?|minutes?|min|h|个月|个星期|个小时|星期|小时|分钟|年|月|周|天|日)(?![A-Za-z])",
    re.IGNORECASE)

# 安全信息 / 治疗建议 / 作用机制：用"标准名称"统一中英文
# 每一项是 (标准名称, 正则)。(?<!副)作用 是为了不把"副作用"算成"作用"
SAFETY_TERMS = [
    ("risk", re.compile(r"\brisks?\b|风险", re.I)),
    ("side_effect", re.compile(r"\bside[ -]?effects?\b|副作用", re.I)),
    ("adverse", re.compile(r"\badverse (?:events?|reactions?|effects?|outcomes?)\b|不良反应|不良事件", re.I)),
    ("contraindication", re.compile(r"\bcontraindicat\w*|禁忌", re.I)),
]
TREATMENT_TERMS = [
    ("recommend", re.compile(r"\brecommend\w*|建议|推荐", re.I)),
    ("treatment", re.compile(r"\btreat\w*|\btherap(?:y|ies|eutic)\b|治疗", re.I)),
    ("regimen", re.compile(r"\bregimens?\b|方案", re.I)),
]
MECHANISM_TERMS = [
    ("mechanism", re.compile(r"\bmechanisms?\b|\bpathways?\b|机制|机理|原理", re.I)),
    ("action", re.compile(r"\bmode of action\b|\bactions?\b|\bacts? (?:on|by|via|through)\b|(?<!副)作用", re.I)),
]

# 六个类别，顺序就是报告里的顺序
CATEGORIES = ["percent", "dosage", "time", "safety", "treatment", "mechanism"]
CATEGORY_NAMES = {
    "percent": "百分比", "dosage": "剂量", "time": "时间范围",
    "safety": "安全信息", "treatment": "治疗建议", "mechanism": "作用机制",
}


def _norm_unit(unit):
    # 按UNIT_MAP换成统一写法；表里没有的就原样（转小写）
    return UNIT_MAP.get(unit.lower(), unit.lower())


def _fmt_number(text):
    # "25.0" -> "25"，"100" -> "100"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_range(first, second):
    # 单个数字返回 "25"，范围返回 "15-25"（中间的 -、~、至、to 都统一成 -）
    first = _fmt_number(first)
    if second:
        return first + "-" + _fmt_number(second)
    return first


def extract_key_info(text):
    """
    用正则从文本里提取六类医学关键信息，返回 {类别: 去重后的标准化字符串集合}。
    数值类的标准化：数字+统一单位，如 "75-100mg"、"12month"、"25%"。
    """
    text = strip_citations(text)
    info = {name: set() for name in CATEGORIES}

    for m in PERCENT_PATTERN.finditer(text):
        info["percent"].add(_fmt_range(m.group(1), m.group(2)) + "%")
    for m in PERCENT_CN_PATTERN.finditer(text):
        info["percent"].add(_fmt_number(m.group(1)) + "%")

    for m in DOSE_PATTERN.finditer(text):
        info["dosage"].add(_fmt_range(m.group(1), m.group(2)) + _norm_unit(m.group(3)))
    # 用药频率没有数值，只要出现就算一条（归到剂量类里），不用finditer
    for pattern, label in FREQ_PATTERNS:
        if pattern.search(text):
            info["dosage"].add(label)

    for m in TIME_PATTERN.finditer(text):
        unit = _norm_unit(m.group(3))
        # 4位数的"年"多半是发表年份（2021年），不是时间范围
        if unit == "year" and len(m.group(1)) == 4 and m.group(2) is None:
            continue
        info["time"].add(_fmt_range(m.group(1), m.group(2)) + unit)

    # 这三类是关键词类：只记"有没有出现"，不管出现几次
    for category, terms in (("safety", SAFETY_TERMS), ("treatment", TREATMENT_TERMS), ("mechanism", MECHANISM_TERMS)):
        for name, pattern in terms:
            if pattern.search(text):
                info[category].add(name)
    return info


def compute_key_info_recall(answer, reference):
    """
    关键信息召回。recall = overlap / gt_matches：
        gt_matches 是标准答案里提取出的关键信息（去重后），overlap 是其中也出现在生成答案里的。
    每个类别单独算一个；标准答案里这个类别一条都没有的，recall记None（没法算）。
    overall是所有类别合在一起的总召回（只算标准答案里有的类别）。
    """
    gt = extract_key_info(reference)
    got = extract_key_info(answer)

    # 只看"标准答案里有的"有多少被覆盖；生成答案多出来的记在extra里，不扣分
    per_category = {}
    total_gt = 0
    total_overlap = 0
    for name in CATEGORIES:
        gt_items = gt[name]
        overlap_items = gt_items & got[name]
        per_category[name] = {
            "gt_matches": sorted(gt_items),
            "answer_matches": sorted(got[name]),
            "overlap": sorted(overlap_items),
            "missed": sorted(gt_items - got[name]),
            "extra": sorted(got[name] - gt_items),
            "recall": (len(overlap_items) / len(gt_items)) if gt_items else None,
        }
        total_gt += len(gt_items)
        total_overlap += len(overlap_items)

    return {
        "per_category": per_category,
        "overall_gt_matches": total_gt,
        "overall_overlap": total_overlap,
        "overall_recall": (total_overlap / total_gt) if total_gt else None,
    }


# ---------------------------------------------------------------------------
# 句子切分（幻觉检测和可读性都要用）
# ---------------------------------------------------------------------------
# 句号后面不应该断开的缩写：先换成占位符，切完再换回来
ABBREVIATIONS = ["e.g.", "i.e.", "vs.", "et al.", "Fig.", "approx.", "No.", "Dr."]


def split_sentences(text):
    """按换行、中文句末标点、英文句号（后面有空格）切句，缩写里的句号不断开"""
    protected = text
    for i, abbr in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"<DOT{i}>"))
    sentences = []
    for line in protected.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 中文句末标点后直接断；英文的 . ! ? 要后面跟空白才断（避免小数点）
        pieces = re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+", line)
        for piece in pieces:
            if piece is None:
                continue
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    result = []
    for s in sentences:
        for i, abbr in enumerate(ABBREVIATIONS):
            s = s.replace(abbr.replace(".", f"<DOT{i}>"), abbr)
        result.append(s)
    return result


def sentence_length(sentence):
    """句子长度：英文单词算1，汉字算1，标点不算"""
    return len(WORD_PATTERN.findall(sentence))


# ---------------------------------------------------------------------------
# 3. 幻觉检测（无依据的绝对化表述）
# ---------------------------------------------------------------------------
# 句子里有这些，认为"有具体引用"
HAS_CITATION_PATTERN = re.compile(r"\[\d+\]|\[来源|et al|\((?:[^()]*\b)?(?:19|20)\d{2}\)|（[^（）]*(?:19|20)\d{2}）")
# 句子里有这些，认为"有限定条件"
QUALIFIER_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"可能", r"部分", r"某些", r"一定程度", r"有限", r"仅", r"只在", r"在[^，。,.]{1,25}(?:患者|人群|条件下|情况下|中)",
        r"\bmay\b", r"\bmight\b", r"\blimited\b", r"\bonly\b", r"\bin (?:\w+ ){0,4}(?:patients|adults|children|trials?|populations?)\b",
        r"\bunder\b", r"\bwhen\b", r"\bamong\b", r"\bsome\b", r"\bcertain\b",
    ]
]

# (信号名, 说明, 正则, 权重, 是否需要"无引用"才触发, 是否需要"无限定条件"才触发)
HALLUCINATION_SIGNALS = [
    {"name": "study_claim_no_citation", "desc": "研究表明 -> 缺少具体引用",
     "pattern": re.compile(r"研究(?:表明|显示|证明|证实|发现)|(?:studies|research|trials?|evidence)\s+(?:have\s+)?"
                           r"(?:show|shown|shows|demonstrate\w*|prove\w*|confirm\w*)", re.I),
     "weight": 1.0, "need_no_citation": True, "need_no_qualifier": False},
    {"name": "proven_no_qualifier", "desc": "已被证明 -> 缺乏限定条件",
     "pattern": re.compile(r"已(?:经)?被?证明|已(?:经)?被?证实|(?:has|have|been|is|are)\s+(?:been\s+)?(?:proven|proved)\b|\bproven to\b", re.I),
     "weight": 1.0, "need_no_citation": False, "need_no_qualifier": True},
    {"name": "hundred_percent", "desc": "100% -> 医学中极少有100%",
     "pattern": re.compile(r"(?<![\d.\-–])100(?:\.0+)?\s*(?:%|％|percent|per cent)|百分之百", re.I),
     "weight": 1.0, "need_no_citation": False, "need_no_qualifier": False},
    {"name": "completely_safe_effective", "desc": "完全(安全|有效|无害) -> 过度绝对化",
     "pattern": re.compile(r"完全(?:安全|有效|无害)|(?:completely|totally|entirely|absolutely)\s+(?:safe|effective|harmless)", re.I),
     "weight": 1.0, "need_no_citation": False, "need_no_qualifier": False},
]
# 额外信号（不在作业要求里，是我补充的）：答案里的数值在给定的上下文里找不到
UNSUPPORTED_NUMBER_WEIGHT = 0.5
UNSUPPORTED_NUMBER_MAX = 2          # 最多算2个，避免一个答案里数字很多时一下子满分

# 风险分 = min(1, 加权信号数 / SCORE_FULL_AT)；分数档位
SCORE_FULL_AT = 3.0
RISK_LOW_BELOW = 0.2
RISK_HIGH_FROM = 0.6


def _context_numbers(context_text):
    # 上下文里出现过的所有数字（整理成统一写法），用来查答案里的数值有没有依据
    return set(_fmt_number(n) for n in re.findall(NUM, context_text))


def detect_hallucination(text, context_text=None):
    """
    检查没有依据的绝对化表述。返回：
        signals        触发的信号列表（信号名、命中的句子、权重）
        weighted_count 加权信号数
        risk_score     0~1，越大越危险
        risk_level     低 / 中 / 高
        citation_markers 正文里的[n]引用标记个数
    context_text 给了的话，会多检查一项：答案里的百分比/剂量/时间数值在上下文里是否找得到。
    """
    sentences = split_sentences(text)
    signals = []

    # 逐句检查：引用和限定条件都要在"同一句"里才算，别的句子里有不算
    for sentence in sentences:
        has_citation = bool(HAS_CITATION_PATTERN.search(sentence))
        has_qualifier = any(p.search(sentence) for p in QUALIFIER_PATTERNS)
        for signal in HALLUCINATION_SIGNALS:
            if not signal["pattern"].search(sentence):
                continue
            if signal["need_no_citation"] and has_citation:
                continue
            if signal["need_no_qualifier"] and has_qualifier:
                continue
            signals.append({"name": signal["name"], "desc": signal["desc"], "weight": signal["weight"],
                            "sentence": sentence[:120]})

    if context_text:
        known = _context_numbers(context_text)
        info = extract_key_info(text)
        unsupported = []
        # 只查百分比/剂量/时间这三类带数字的；一个条目里只要有一个数字在上下文里找不到就算
        for category in ("percent", "dosage", "time"):
            for item in sorted(info[category]):
                numbers = [_fmt_number(n) for n in re.findall(NUM, item)]
                if numbers and any(n not in known for n in numbers):
                    unsupported.append(item)
        for item in unsupported[:UNSUPPORTED_NUMBER_MAX]:
            signals.append({"name": "number_not_in_context", "desc": "数值在上下文里找不到（额外信号）",
                            "weight": UNSUPPORTED_NUMBER_WEIGHT, "sentence": item})

    weighted = sum(s["weight"] for s in signals)
    score = min(1.0, weighted / SCORE_FULL_AT)
    if score < RISK_LOW_BELOW:
        level = "低"
    elif score < RISK_HIGH_FROM:
        level = "中"
    else:
        level = "高"
    return {
        "signals": signals,
        "signal_count": len(signals),
        "weighted_count": weighted,
        "risk_score": score,
        "risk_level": level,
        "citation_markers": len(re.findall(r"\[\d+\]|\[来源[:：]", text)),
        "context_checked": bool(context_text),
    }


# ---------------------------------------------------------------------------
# 4. 可读性
# ---------------------------------------------------------------------------
IDEAL_LEN_LOW = 10          # 平均句长在 10~25 之间认为最容易读
IDEAL_LEN_HIGH = 25
ZERO_AT_LEN = 60            # 平均句长到60分数降到0
LONG_SENTENCE_LEN = 40      # 超过40算长句


def compute_readability(text):
    """可读性：句子数、平均句长、最长句、长句比例、段落数、英文平均词长，以及一个0~1的启发式分数"""
    sentences = split_sentences(text)
    lengths = [sentence_length(s) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    english_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)

    if not lengths:
        return {"sentence_count": 0, "avg_sentence_length": 0.0, "max_sentence_length": 0,
                "long_sentence_ratio": 0.0, "paragraph_count": len(paragraphs),
                "avg_english_word_length": 0.0, "readability_score": 0.0}

    avg = sum(lengths) / len(lengths)
    long_ratio = sum(1 for n in lengths if n > LONG_SENTENCE_LEN) / len(lengths)
    if avg < IDEAL_LEN_LOW:
        length_score = avg / IDEAL_LEN_LOW
    elif avg <= IDEAL_LEN_HIGH:
        length_score = 1.0
    else:
        length_score = max(0.0, 1 - (avg - IDEAL_LEN_HIGH) / (ZERO_AT_LEN - IDEAL_LEN_HIGH))
    score = length_score * (1 - 0.5 * long_ratio)

    avg_word = (sum(len(w) for w in english_words) / len(english_words)) if english_words else 0.0
    return {
        "sentence_count": len(lengths),
        "avg_sentence_length": avg,
        "max_sentence_length": max(lengths),
        "long_sentence_ratio": long_ratio,
        "paragraph_count": len(paragraphs),
        "avg_english_word_length": avg_word,
        "readability_score": score,
    }


# ---------------------------------------------------------------------------
# 评估器
# ---------------------------------------------------------------------------
class AnswerEvaluator:
    """把上面几个检查（ROUGE、关键信息召回、幻觉检测、可读性）合在一起，对一个答案出一份报告"""

    def __init__(self, rouge_backend="auto"):
        """rouge_backend: "auto"（有rouge库用库，没有用自己写的）/ "rouge" / "builtin" """
        self.rouge_backend = rouge_backend

    def evaluate(self, answer, reference=None, context_text=None, body_only=True):
        """
        评估一个答案。
        Args:
            answer: 生成的答案
            reference: 标准答案。不给的话只做幻觉检测和可读性（相似性和关键信息召回需要标准答案）
            context_text: 给模型的上下文。给了的话幻觉检测会多查一项"数值在不在上下文里"
            body_only: True就先去掉参考文献和免责声明（评估正文）
        """
        # 标准答案也按同样方式去掉尾部，两边比的才是同样的东西
        text = get_answer_body(answer) if body_only else answer
        reference_text = get_answer_body(reference) if (reference and body_only) else reference

        report = {
            "answer_chars": len(answer),
            "body_chars": len(text),
            "similarity": None,
            "key_info": None,
            # 幻觉检测要看有没有[1]这样的引用，所以用带引用标记的原文；可读性算句长，去掉标记更准
            "hallucination": detect_hallucination(text, context_text),
            "readability": compute_readability(strip_citations(text)),
        }
        if reference_text:
            report["similarity"] = compute_rouge(text, reference_text, backend=self.rouge_backend)
            report["key_info"] = compute_key_info_recall(text, reference_text)
        return report

    def evaluate_result(self, result, reference=None):
        """评估MedicalGenerationPipeline.generate()返回的结果字典"""
        # 取的是result里保存的 context_text（组装后的上下文），不是筛选之后的 context_text_filtered
        context_text = None
        inter = result.get("intermediate_results") or {}
        if inter.get("context_text"):
            context_text = inter["context_text"]
        return self.evaluate(result["answer"], reference=reference, context_text=context_text)


def format_evaluation_report(report, title=None):
    """把evaluate()返回的报告字典整理成一段文字（返回字符串）。title不为空时放在最前面当标题"""
    lines = []
    if title:
        lines.append(f"--- {title} ---")
    lines.append(f"正文长度：{report['body_chars']}个字符（整个答案{report['answer_chars']}个字符）")

    sim = report["similarity"]
    if sim is None:
        lines.append("文本相似性：（没有标准答案，跳过）")
    else:
        lines.append(f"文本相似性（ROUGE，后端={sim['backend']}，词数 生成{sim['hyp_len']}/标准{sim['ref_len']}）：")
        for name in ("rouge-1", "rouge-2", "rouge-l"):
            s = sim[name]
            lines.append(f"    {name:<8} 召回={s['r']:.3f}  精确={s['p']:.3f}  F1={s['f']:.3f}")
        if sim.get("language_mismatch"):
            lines.append(f"    注意：生成答案和标准答案的语言差别很大（中文字符占比 {sim['hyp_cjk_ratio']:.0%} vs {sim['ref_cjk_ratio']:.0%}），"
                         f"ROUGE基本只能匹配到数字和少数专有名词，分数偏低不代表答案不好；关键信息召回已做中英文统一，不受这个影响。")

    key = report["key_info"]
    if key is None:
        lines.append("关键信息召回：（没有标准答案，跳过）")
    else:
        overall = key["overall_recall"]
        overall_text = "无法计算(标准答案没有关键信息)" if overall is None else f"{overall:.3f}"
        lines.append(f"关键信息召回：总召回={overall_text}  "
                     f"（覆盖{key['overall_overlap']}/{key['overall_gt_matches']}条）")
        for name in CATEGORIES:
            c = key["per_category"][name]
            if c["recall"] is None:
                lines.append(f"    {CATEGORY_NAMES[name]:<6} 标准答案里没有，不计")
            else:
                missed = f"  漏掉={c['missed']}" if c["missed"] else ""
                lines.append(f"    {CATEGORY_NAMES[name]:<6} {c['recall']:.2f} ({len(c['overlap'])}/{len(c['gt_matches'])}){missed}")

    h = report["hallucination"]
    lines.append(f"幻觉风险：{h['risk_level']}（风险分={h['risk_score']:.2f}，触发信号{h['signal_count']}个，"
                 f"引用标记{h['citation_markers']}个" + ("，已对照上下文数值" if h["context_checked"] else "") + "）")
    for s in h["signals"]:
        lines.append(f"    - [{s['name']}] {s['desc']}：{s['sentence']}")

    r = report["readability"]
    lines.append(f"可读性：{r['readability_score']:.2f}  句子{r['sentence_count']}个，平均句长{r['avg_sentence_length']:.1f}，"
                 f"最长{r['max_sentence_length']}，长句比例{r['long_sentence_ratio']:.0%}，段落{r['paragraph_count']}个，"
                 f"英文平均词长{r['avg_english_word_length']:.1f}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 直接运行：用两段短文本演示一遍（不需要任何模型）
    reference = "Metformin reduced myocardial infarction by 39% in the UKPDS trial. Common side effects include GI upset."
    good = "二甲双胍在UKPDS研究中使心肌梗死降低39%[1]。常见副作用是胃肠道不适。"
    bad = "研究表明二甲双胍100%安全，已被证明完全有效。"
    evaluator = AnswerEvaluator()
    print(format_evaluation_report(evaluator.evaluate(good, reference), "较好的答案"))
    print()
    print(format_evaluation_report(evaluator.evaluate(bad, reference), "夸大的答案"))
