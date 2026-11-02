"""
用真实的tokenizer验证上下文组装器的token估算和截断

之前在沙箱里测的时候用的是假tokenizer（大概每个词算1~2个token），所以token数
是不是准、截断会不会超预算，都还没有在真tokenizer上验证过。这个脚本在本机运行，
加载真正的bge-reranker-base的tokenizer，检查下面几件事：

    检查1 tokenizer能不能正常加载、几段典型文本的token数是多少
    检查2 截断用的二分查找要求"前缀越长token数越多"，真tokenizer上是不是成立
    检查3 不同预算下，截断后的token数会不会超预算、是不是停在句号处
    检查4 （可选）如果本机开着Ollama，对比一下我们估算的token数和qwen3:8b真实算出来的token数

用法：python 18_verify_tokenizer.py
没有命令行参数。输入用的是17_demo_context_assembler.py里的内置示例文档（EXAMPLE_DOCS）；
输出是每项检查的 [PASS]/[FAIL]/[WARN]，最后汇总三种结果的个数，不写文件。
第一次运行要能下载（或本地已缓存）bge-reranker-base的tokenizer；检查4连不上Ollama会自动跳过，不算失败。
"""

import importlib
import json
import urllib.request

_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
_m_demo_context_assembler = importlib.import_module("17_demo_context_assembler")
EXAMPLE_DOCS = _m_demo_context_assembler.EXAMPLE_DOCS
build_docs = _m_demo_context_assembler.build_docs

# 检查3里要试的几个预算
TEST_BUDGETS = [30, 60, 120, 200, 500]

# 检查2里，前缀token数回落的幅度不超过这个数就认为二分查找可以接受
MAX_ALLOWED_PREFIX_DROP = 3

# Ollama的地址和模型名（检查4用，跟以后接生成模块要用的一致）
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# 估算值和Ollama真实值相差超过这个比例就给警告
OLLAMA_DIFF_WARN_RATIO = 0.15

pass_count = 0
fail_count = 0
warn_count = 0


def check(name, ok, detail=""):
    """记录并打印一项检查，ok为True算PASS，否则算FAIL"""
    global pass_count, fail_count
    if ok:
        pass_count += 1
        print(f"  [PASS] {name} {detail}")
    else:
        fail_count += 1
        print(f"  [FAIL] {name} {detail}")


def warn(name, detail=""):
    """记录并打印一条警告（不算失败，需要人工看一下）"""
    global warn_count
    warn_count += 1
    print(f"  [WARN] {name} {detail}")


def check_basic(assembler):
    """检查1：tokenizer能正常数token（空串是0、英文句子大于0、带连字符的术语会被拆成子词）"""
    print("\n--- 检查1 tokenizer基本情况 ---")
    samples = [
        ("空字符串", ""),
        ("一句英文医学文本", "Metformin reduces cardiovascular events in patients with type 2 diabetes."),
        ("带连字符和数字的术语", "HMG-CoA reductase inhibitors lowered LDL-C by 2.5 mmol/L (p<0.001)."),
        ("一句中文", "二甲双胍可以降低2型糖尿病患者的心血管事件风险。"),
    ]
    for name, text in samples:
        n = assembler.estimate_tokens(text)
        words = len(text.split())
        print(f"  {name}: {len(text)}个字符，{words}个空格分词，真tokenizer算出{n}个token")

    check("空字符串是0个token", assembler.estimate_tokens("") == 0)
    check("英文句子的token数大于0", assembler.estimate_tokens(samples[1][1]) > 0)

    # 医学术语一般会被拆成好几个子词，token数应该比空格分出的词数多
    term_text = samples[2][1]
    check("带连字符的术语被拆成了子词（token数 > 词数）",
          assembler.estimate_tokens(term_text) > len(term_text.split()),
          f"(token={assembler.estimate_tokens(term_text)}, 词={len(term_text.split())})")


def check_prefix_monotonic(assembler, full_context):
    """检查2：把完整上下文的每个前缀都数一遍token，看有没有变少的地方（二分查找要求前缀越长token数不减少）"""
    # _cut_to_token_limit用二分查找，前提是"前缀越长，token数不会变少"。
    # 子词tokenizer在个别位置可能不严格满足（比如一个词切到一半），这里把每个前缀都数一遍
    print("\n--- 检查2 前缀token数是否单调不减（二分查找的前提）---")
    prev = 0
    violations = []
    max_drop = 0
    for n in range(len(full_context) + 1):
        t = assembler.estimate_tokens(full_context[:n])
        if t < prev:
            violations.append((n, prev, t))
            max_drop = max(max_drop, prev - t)
        prev = t
    if not violations:
        check("每个前缀的token数都没有比前一个前缀少", True, f"(共检查{len(full_context) + 1}个前缀)")
    else:
        # 真tokenizer上这种回落很常见：一个词只切到一半时会被拆成好几个碎片，
        # 补全成完整的词之后碎片又合并成一个token，token数就少了1~2个。
        # 对二分查找的影响：找到的位置仍然保证不超预算（只有验证过<=预算的位置才会被采用），
        # 最多是比理论上的最长前缀短几个字符。所以只要回落幅度很小就算通过
        detail = (f"(共{len(violations)}处回落，最大回落{max_drop}个token，"
                  f"前3处(位置, 前一个, 当前): {violations[:3]})")
        if max_drop <= MAX_ALLOWED_PREFIX_DROP:
            check(f"前缀token数有小幅回落，但幅度不超过{MAX_ALLOWED_PREFIX_DROP}，二分查找可以接受", True, detail)
        else:
            warn(f"前缀token数回落幅度超过{MAX_ALLOWED_PREFIX_DROP}，二分查找的结果可能偏差较大", detail)


def check_budgets(assembler, docs):
    """检查3：用TEST_BUDGETS里的每个预算调一遍assemble()，看有没有超预算、是不是停在句号处、预算是不是用得太少"""
    print("\n--- 检查3 不同预算下的截断结果 ---")
    # 先算一下不截断时完整上下文有多少token（预算给得很大就不会截断）
    full_tokens = assembler.assemble(docs, max_context_tokens=10 ** 6)["metadata"]["estimated_tokens"]
    print(f"  不截断时完整上下文共{full_tokens}个token")

    for budget in TEST_BUDGETS:
        result = assembler.assemble(docs, max_context_tokens=budget)
        text = result["context_text"]
        tokens = result["metadata"]["estimated_tokens"]
        ends_ok = text.rstrip()[-1:] in ".。!?！？"
        used_ratio = tokens / budget
        truncated = full_tokens > budget
        print(f"  预算{budget}: 实际{tokens}个token（用了{used_ratio:.0%}），"
              f"{'发生了截断' if truncated else '没超预算，没有截断'}，结尾{'是' if ends_ok else '不是'}句号")
        check(f"预算{budget}：没有超预算", tokens <= budget, f"({tokens} <= {budget})")
        if text:
            if ends_ok:
                check(f"预算{budget}：停在句子结束处", True)
            else:
                warn(f"预算{budget}：没有停在句号处", f"结尾是: ...{text[-30:]!r}")
        # 发生了截断、但预算用得太少，说明截得太狠，浪费预算（没截断的话内容本来就没那么多，不算）
        if truncated and used_ratio < 0.5 and tokens > 0:
            warn(f"预算{budget}：只用了{used_ratio:.0%}，截得偏狠")


def check_against_ollama(assembler, docs):
    """检查4（可选）：对比bge tokenizer估的token数和Ollama里qwen3:8b真实的token数；连不上Ollama就跳过"""
    # 让Ollama自己数一遍同一段文本有多少个token，跟我们用bge tokenizer估的对比。
    # raw=True是不套对话模板，num_predict=1是只生成1个token，这样返回的
    # prompt_eval_count基本就是输入文本本身的token数
    print("\n--- 检查4 对比Ollama里qwen3:8b真实的token数（可选）---")
    text = assembler.assemble(docs, max_context_tokens=500)["context_text"]
    our_count = assembler.estimate_tokens(text)

    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": text,
        "raw": True,
        "stream": False,
        "options": {"num_predict": 1},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [跳过] 连不上Ollama或者模型没拉取（{type(e).__name__}: {e}）")
        print(f"         想做这项检查的话：先 ollama serve，并且 ollama pull {OLLAMA_MODEL}")
        return

    real_count = data.get("prompt_eval_count")
    if real_count is None:
        print("  [跳过] Ollama没有返回prompt_eval_count（提示词被缓存时可能会这样），再运行一次试试")
        return

    diff_ratio = abs(real_count - our_count) / real_count if real_count else 0
    print(f"  同一段上下文：bge tokenizer估算={our_count}，qwen3:8b真实={real_count}，"
          f"相差{diff_ratio:.0%}")
    if diff_ratio <= OLLAMA_DIFF_WARN_RATIO:
        check("估算值跟qwen3:8b真实值相差不大", True, f"(相差{diff_ratio:.0%})")
    else:
        warn(f"估算值跟qwen3:8b真实值相差{diff_ratio:.0%}，超过{int(OLLAMA_DIFF_WARN_RATIO * 100)}%",
             "预算设置的时候要留余量，或者以后换成qwen自己的tokenizer来估算")
    if real_count > our_count:
        print("  注意：真实值比估算值大，说明按估算值卡预算的话，实际送进模型的可能会略超预算")


if __name__ == "__main__":
    assembler = ContextAssembler()  # 不传tokenizer，用真的bge-reranker-base

    docs = build_docs(EXAMPLE_DOCS)

    check_basic(assembler)

    # 检查2用内置示例拼出来的完整上下文（这里自己照着assemble()的前几步拼一遍，只拼不截断，因为要的是没截断的完整文本）
    chunks = [assembler._to_document_chunk(d) for d in docs]
    selected = assembler._select_diverse_chunks(assembler._deduplicate(chunks))
    full_context = "\n\n".join(f"[来源: {c.source}]\n{c.text}" for c in selected)
    check_prefix_monotonic(assembler, full_context)

    check_budgets(assembler, docs)
    check_against_ollama(assembler, docs)

    print("\n" + "=" * 60)
    print(f"汇总：PASS={pass_count}  FAIL={fail_count}  WARN={warn_count}")
