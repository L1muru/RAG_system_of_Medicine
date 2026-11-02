"""
上下文组装器的逐步演示/自我验证脚本

输入一批文档（可以用内置示例，也可以自己手动输入），脚本会把上下文组装器
（14_context_assembler.py）的每一步单独拆出来，把"这一步的输入是什么、输出是什么、
为什么这么处理"打印出来，方便自己对照检查每一步是不是符合要求：

    步骤1 转换文档格式（字典 -> DocumentChunk）
    步骤2 计算文本相似性并去重（Jaccard）
    步骤3 按照相关性排序
    步骤4 优先选择高相关性文档，并考虑多样化（同一来源太多就降低优先级）
    步骤5 构建上下文字符串并限制长度（先在后10%文本里找句号，找不到放宽到后50%，在完整句子处截断）

最后再调用一遍assemble()，检查它的结果跟我们逐步拆开算出来的结果是不是一致。

用法：python 17_demo_context_assembler.py
没有命令行参数，运行后是交互式菜单：输入1用内置示例，输入2手动输入文档（每行"来源 | 相关性分数 | 文本"，
空行结束），输入q退出；选完之后再问一个token预算（直接回车用默认值）。
输出：只是把每一步的过程打印到屏幕上，不会写文件。
运行时会创建ContextAssembler()，不传tokenizer，所以会加载bge-reranker-base的tokenizer。
"""
import importlib

_m_context_assembler = importlib.import_module("14_context_assembler")
ContextAssembler = _m_context_assembler.ContextAssembler
JACCARD_DEDUP_THRESHOLD = _m_context_assembler.JACCARD_DEDUP_THRESHOLD
MAX_CHUNKS_PER_SOURCE = _m_context_assembler.MAX_CHUNKS_PER_SOURCE
TRUNCATE_SEARCH_WINDOW_RATIO = _m_context_assembler.TRUNCATE_SEARCH_WINDOW_RATIO
TRUNCATE_FALLBACK_WINDOW_RATIO = _m_context_assembler.TRUNCATE_FALLBACK_WINDOW_RATIO

# 演示时默认的token预算，故意设得比较小，这样内置示例会触发截断，能看到截断这一步的效果
DEFAULT_DEMO_MAX_TOKENS = 120

# 内置示例：(来源, 相关性分数, 文本)。故意设计成：
# - 输入顺序没有按相关性排好（用来看排序）
# - Paper B和Paper C的内容几乎一样，只差一个词（用来看去重）
# - Paper A有4条（用来看同一来源太多时的降级）
# - 总长度超过预算（用来看截断）
EXAMPLE_DOCS = [
    ("Paper A", 0.62, "Aspirin inhibits platelet aggregation by irreversibly blocking cyclooxygenase enzymes. "
                      "It is widely used for secondary prevention of cardiovascular events."),
    ("Paper B", 0.95, "Metformin reduces cardiovascular events in patients with type 2 diabetes by activating AMPK "
                      "and improving insulin sensitivity. Large trials have supported this benefit."),
    ("Paper A", 0.80, "Statins reduce LDL cholesterol through inhibition of HMG-CoA reductase in the liver. "
                      "This lowers the long term risk of myocardial infarction."),
    ("Paper C", 0.90, "Metformin reduces cardiovascular events in patients with type 2 diabetes through activating AMPK "
                      "and improving insulin sensitivity. Large trials have supported this benefit."),
    ("Paper A", 0.70, "Warfarin dosing is adjusted according to international normalized ratio measurements. "
                      "Regular monitoring is required to balance bleeding and clotting risk."),
    ("Paper A", 0.55, "Lisinopril lowers blood pressure by inhibiting the angiotensin converting enzyme pathway. "
                      "It is commonly prescribed for hypertension and heart failure."),
    ("Paper D", 0.40, "Insulin therapy requires careful glucose monitoring to avoid hypoglycemia in elderly patients. "
                      "Dose adjustment should follow clinical guidance."),
]


def preview(text, n=60):
    """把文本里的换行换成空格，超过n个字符就截断加省略号，打印的时候看着短一点"""
    text = text.replace("\n", " ")
    if len(text) <= n:
        return text
    return text[:n] + "..."


def build_docs(rows):
    # 把(来源, 分数, 文本)转成跟12_reranker.py输出格式一样的字典
    docs = []
    for i, (source, score, text) in enumerate(rows):
        docs.append({
            "chunk_id": f"c{i + 1}",
            "text": text,
            "metadata": {"source_title": source},
            "final_score": score,
        })
    return docs


def step1_convert(assembler, docs):
    """步骤1：把文档字典转成DocumentChunk并打印，返回DocumentChunk列表"""
    print("\n【步骤1 转换文档格式】把输入的字典转成DocumentChunk")
    chunks = [assembler._to_document_chunk(d) for d in docs]
    for c in chunks:
        print(f"  {c.chunk_id}  来源={c.source}  相关性={c.relevance_score}  文本={preview(c.text)}")
    return chunks


def step2_dedup(assembler, chunks):
    """步骤2：自己按相关性从高到低做一遍Jaccard去重并打印每条的去留，再跟模块的_deduplicate()对比。返回保留下来的chunk"""
    print(f"\n【步骤2 Jaccard去重】阈值={JACCARD_DEDUP_THRESHOLD}，"
          f"按相关性从高到低，依次跟已经保留下来的chunk比较")
    ordered = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    kept = []
    # 这里的逻辑是照着模块里的_deduplicate()自己重写了一遍（先排序，再跟已保留的比），
    # 目的是把每一条为什么留、为什么丢打印出来；最后会跟模块的结果对比
    for c in ordered:
        best_sim = 0.0
        best_id = None
        for k in kept:
            sim = assembler._jaccard_similarity(c.text, k.text)
            if sim > best_sim:
                best_sim = sim
                best_id = k.chunk_id
        if best_sim >= JACCARD_DEDUP_THRESHOLD:
            print(f"  {c.chunk_id}(相关性{c.relevance_score})  与已保留的{best_id}相似度={best_sim:.2f} "
                  f">= 阈值  -> 丢弃（重复）")
        elif kept:
            print(f"  {c.chunk_id}(相关性{c.relevance_score})  与已保留chunk的最大相似度={best_sim:.2f}  -> 保留")
            kept.append(c)
        else:
            print(f"  {c.chunk_id}(相关性{c.relevance_score})  第一条  -> 保留")
            kept.append(c)

    module_result = [c.chunk_id for c in assembler._deduplicate(chunks)]
    same = [c.chunk_id for c in kept] == module_result
    print(f"  去重前{len(chunks)}条，去重后{len(kept)}条。跟模块里_deduplicate()的结果一致: {'是' if same else '否(有问题!)'}")
    return kept


def step3_sort(chunks):
    """步骤3：按相关性从高到低排序并打印名次，返回排好序的列表（这个步骤的返回值在run_demo里没有用到）"""
    print("\n【步骤3 按照相关性排序】从高到低")
    ordered = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    for i, c in enumerate(ordered):
        print(f"  第{i + 1}名  {c.chunk_id}  相关性={c.relevance_score}  来源={c.source}")
    return ordered


def step4_diversify(assembler, chunks):
    """步骤4：同一来源超过上限的chunk降级排到后面，打印每条的处理，并跟_select_diverse_chunks()对比。返回最终顺序"""
    print(f"\n【步骤4 多样化选择】同一来源最多{MAX_CHUNKS_PER_SOURCE}条正常优先级，"
          f"超过的降级排到后面")
    ordered = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    counts = {}
    normal = []
    lowered = []
    for c in ordered:
        n = counts.get(c.source, 0)
        if n < MAX_CHUNKS_PER_SOURCE:
            normal.append(c)
            counts[c.source] = n + 1
            print(f"  {c.chunk_id}  来源={c.source}（已选{n}条）  -> 正常优先级")
        else:
            lowered.append(c)
            print(f"  {c.chunk_id}  来源={c.source}（已选{n}条，已达上限）  -> 降级")

    final_order = normal + lowered
    print("  最终顺序:", " > ".join(f"{c.chunk_id}({c.source})" for c in final_order))

    module_result = [c.chunk_id for c in assembler._select_diverse_chunks(chunks)]
    same = [c.chunk_id for c in final_order] == module_result
    print(f"  跟模块里_select_diverse_chunks()的结果一致: {'是' if same else '否(有问题!)'}")
    return final_order


def step5_build_and_truncate(assembler, selected, max_tokens):
    """步骤5：拼接上下文字符串，超过token预算就截断（先二分找前缀，再退到句号），并跟_truncate_at_sentence_boundary()对比。返回最终上下文"""
    print(f"\n【步骤5 构建上下文字符串并限制长度】token预算={max_tokens}，"
          f"截断时先在最后{int(TRUNCATE_SEARCH_WINDOW_RATIO * 100)}%的文本里找句号，"
          f"找不到再放宽到最后{int(TRUNCATE_FALLBACK_WINDOW_RATIO * 100)}%")
    pieces = [f"[来源: {c.source}]\n{c.text}" for c in selected]
    full_context = "\n\n".join(pieces)
    full_tokens = assembler.estimate_tokens(full_context)
    print(f"  拼接出来的完整上下文: {len(full_context)}个字符，约{full_tokens}个token")

    if full_tokens <= max_tokens:
        final_context = full_context
        print(f"  没超过预算（{full_tokens} <= {max_tokens}），不需要截断")
    else:
        # 把截断拆成两小步，方便看每一步做了什么
        prefix = assembler._cut_to_token_limit(full_context, max_tokens)
        prefix_tokens = assembler.estimate_tokens(prefix)
        print(f"  超过预算（{full_tokens} > {max_tokens}），先二分查找出预算内最长的前缀："
              f"{len(prefix)}个字符，{prefix_tokens}个token")
        final_context, ratio = assembler._trim_to_sentence_end(prefix)

        if ratio is None:
            print("  最后50%范围里也没有句号，只好保留前缀原样（会停在句子中间）")
        elif ratio == TRUNCATE_SEARCH_WINDOW_RATIO:
            print(f"  在前缀的最后{int(ratio * 100)}%里找到了句号，从这里截断（符合任务要求）")
        else:
            print(f"  最后{int(TRUNCATE_SEARCH_WINDOW_RATIO * 100)}%里没有句号，"
                  f"放宽到最后{int(ratio * 100)}%才找到（这是额外加的兜底）")

        dropped = full_context[len(final_context):]
        print(f"  最后保留{len(final_context)}个字符，丢掉后面{len(dropped)}个字符")
        print(f"  截断后的结尾: ...{final_context[-50:]!r}")
        print(f"  被丢掉部分的开头: {dropped.strip()[:50]!r}...")

        # 跟模块里完整的截断函数对一下
        same = final_context == assembler._truncate_at_sentence_boundary(full_context, max_tokens)
        print(f"  跟模块里_truncate_at_sentence_boundary()的结果一致: {'是' if same else '否(有问题!)'}")

    final_tokens = assembler.estimate_tokens(final_context)
    ends_ok = final_context.rstrip()[-1:] in ".。!?！？"
    print(f"  结尾是不是句号: {'是' if ends_ok else '否'}")
    print(f"  最终上下文token数: {final_tokens}（预算{max_tokens}，"
          f"{'没超' if final_tokens <= max_tokens else '超了(有问题!)'}）")
    return final_context


def run_demo(assembler, docs, max_tokens):
    """对一批文档依次跑步骤1~5，最后再调assemble()对照结果是否一致"""
    print("\n" + "=" * 60)
    print(f"输入共{len(docs)}条文档，token预算={max_tokens}")
    print("=" * 60)

    chunks = step1_convert(assembler, docs)
    unique_chunks = step2_dedup(assembler, chunks)
    step3_sort(unique_chunks)
    selected = step4_diversify(assembler, unique_chunks)
    final_context = step5_build_and_truncate(assembler, selected, max_tokens)

    print("\n【最终上下文】")
    print("-" * 40)
    print(final_context)
    print("-" * 40)

    # 最后调用一遍assemble()，检查它的结果跟我们逐步拆开算出来的是不是一致
    result = assembler.assemble(docs, max_context_tokens=max_tokens)
    print("\n【assemble()的返回结果】")
    print("  metadata:", result["metadata"])
    same_context = result["context_text"] == final_context
    same_selected = [c.chunk_id for c in result["selected_chunks"]] == [c.chunk_id for c in selected]
    print(f"  上下文文本跟逐步计算的一致: {'是' if same_context else '否(有问题!)'}")
    print(f"  选中的chunk及顺序跟逐步计算的一致: {'是' if same_selected else '否(有问题!)'}")


def read_custom_docs():
    """从键盘读用户手动输入的文档，返回[(来源, 分数, 文本)]；格式不对的行会跳过"""
    print("\n请输入文档，每行一条，格式：来源 | 相关性分数 | 文本")
    print("例如：Paper A | 0.85 | Metformin reduces cardiovascular events in diabetic patients.")
    print("输入完成后，直接回车（输入空行）结束。")
    rows = []
    while True:
        line = input("> ").strip()
        if line == "":
            break
        # 最多切成3段（split的第二个参数是2），这样文本里自己带 | 也不会被切碎
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            print("  格式不对，需要用两个 | 分成三段，这一行已跳过")
            continue
        source, score_text, text = parts
        try:
            score = float(score_text)
        except ValueError:
            print("  相关性分数不是数字，这一行已跳过")
            continue
        rows.append((source, score, text))
    return rows


def ask_max_tokens():
    """问用户要token预算，回车或者输入的不是整数就用默认值"""
    text = input(f"token预算（直接回车用默认值{DEFAULT_DEMO_MAX_TOKENS}）: ").strip()
    if text == "":
        return DEFAULT_DEMO_MAX_TOKENS
    try:
        return int(text)
    except ValueError:
        print("不是整数，使用默认值")
        return DEFAULT_DEMO_MAX_TOKENS


if __name__ == "__main__":
    assembler = ContextAssembler()

    print("\n上下文组装器逐步演示，输入q退出")
    while True:
        choice = input("\n请选择：1=使用内置示例  2=手动输入文档  q=退出: ").strip()
        if choice == "q":
            break
        if choice == "1":
            rows = EXAMPLE_DOCS
        elif choice == "2":
            rows = read_custom_docs()
            if not rows:
                print("没有输入有效的文档")
                continue
        else:
            print("请输入1、2或q")
            continue

        max_tokens = ask_max_tokens()
        run_demo(assembler, build_docs(rows), max_tokens)
