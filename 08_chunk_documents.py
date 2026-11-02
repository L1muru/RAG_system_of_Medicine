"""
文本分块脚本（整个流程的第08步）

做什么：把解析好的PubMed论文切成一个个文本块（chunk），给后面建向量索引用。
输入：
    pkl_model/parsed_articles.pkl（01_parse_articles.py产出，要用到body_text、abstract、
    pmid、pmcid、title、filename这几列）
    medrag_data/pubmed_raw/extracted/oa_comm_aws/ 下的原始XML（00_download_oa_comm.py下载的），
    按filename列一篇一篇重新读，按<sec>章节取正文
产出（都在chunks_output/目录下）：
    chunks.parquet、chunks.jsonl：最终的chunk数据集（chunk_id、text、doc_id、chunk_index、
    total_chunks、source_title、section_title、token_count）
    processing_stats.json：这次分块的参数和统计信息
    dropped_short_chunks.csv：被当成低价值短块剔除的chunk（有被剔除的才会生成）
    skipped_documents.csv：XML解析失败被跳过的文献（有才会生成）
大致流程：清洗并生成doc_id -> 按章节提取正文（去掉公式、表格换成占位描述）
    -> 章节不超过TOKEN_LIMIT就整段当一个chunk，超过就用滑动窗口切 -> 过滤短块
    -> 保存 -> 打印一些质量检查结果

运行方式：python 08_chunk_documents.py
    （不需要命令行参数，第一次运行会下载tokenizer，需要能联网或本地已有缓存）
下一步：09_build_vector_index.py 读取chunks.parquet去建向量索引
依赖：pandas、lxml、transformers、langchain_text_splitters（保存parquet还需要pyarrow之类的引擎）
"""

import os
import json
import pandas as pd
from lxml.etree import parse, XMLParser, strip_elements
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "medrag_data", "pubmed_raw", "extracted", "oa_comm_aws")
PKL_DIR = os.path.join(BASE_DIR, "pkl_model")
OUTPUT_DIR = os.path.join(BASE_DIR, "chunks_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
TOKEN_LIMIT = 512  # 嵌入模型（NeuML/pubmedbert-base-embeddings）上限

FORMULA_TAGS = ["tex-math", "{http://www.w3.org/1998/Math/MathML}math"]
xml_parser = XMLParser(recover=True)

print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")


def count_tokens(text):
    """用上面加载的tokenizer数一段文本有多少个token，不截断。参数text是字符串，返回int"""
    return len(tokenizer.encode(text, truncation=False))


# ========== 1. 环境准备与数据加载 ==========
print("\n===== 1. 数据加载与基础清洗 =====")

# 用最原始的parsed_articles.pkl做输入：本脚本自己负责清洗和doc_id生成，
# 不依赖02_check_quality.py是否已经把清洗结果写回pkl，避免耦合不确定的中间产物。
input_path = os.path.join(PKL_DIR, "parsed_articles.pkl")
df_raw = pd.read_pickle(input_path)
print(f"原始记录数: {len(df_raw)}")

# 基础清洗：正文为空/全空白的记录无法生成任何chunk，剔除
before = len(df_raw)
df_raw = df_raw[df_raw["body_text"].notna() & (df_raw["body_text"].str.strip() != "")].reset_index(drop=True)
print(f"基础清洗（剔除正文为空）：{before} -> {len(df_raw)}")


# 摘要缺失/占位符的填充策略（沿用上周设计说明1.3节结论）：
# 用body_text开头若干句填充，而不是丢弃
def fill_abstract(row):
    """摘要为空或是占位符"xx"时，用body_text前500个字符代替；否则原样返回摘要"""
    abstract = row.get("abstract")
    if pd.isna(abstract) or not str(abstract).strip() or str(abstract).strip() == "xx":
        body = row.get("body_text") or ""
        return body[:500]  # 取正文开头一段代替摘要
    return abstract


df_raw["abstract"] = df_raw.apply(fill_abstract, axis=1)


# 生成/确认文献唯一标识 doc_id：优先pmid，缺失时用pmcid兜底，两者都没有再生成uuid
def make_doc_id(row):
    """给一行文献生成doc_id：有pmid用pmid_xxx，没有用pmcid_xxx，都没有用uuid_行号"""
    pmid = row.get("pmid")
    if pd.notna(pmid) and str(pmid).strip():
        return f"pmid_{str(pmid).strip()}"
    pmcid = row.get("pmcid")
    if pd.notna(pmcid) and str(pmcid).strip():
        return f"pmcid_{str(pmcid).strip()}"
    return f"uuid_{row.name}"


df_raw["doc_id"] = df_raw.apply(make_doc_id, axis=1)

dup_count = df_raw["doc_id"].duplicated().sum()
if dup_count:
    print(f"⚠️ 发现{dup_count}个重复doc_id，追加序号去重")
    dup_suffix = df_raw.groupby("doc_id").cumcount()
    df_raw.loc[dup_suffix > 0, "doc_id"] = (
            df_raw.loc[dup_suffix > 0, "doc_id"] + "_dup" + dup_suffix[dup_suffix > 0].astype(str)
    )

assert df_raw["doc_id"].is_unique, "doc_id生成后仍存在重复，需要检查make_doc_id逻辑"
print(f"doc_id生成完成，唯一性校验通过，共{len(df_raw)}篇文献")

# ========== 2. 按<sec>章节边界提取正文结构 ==========
TABLE_TAGS = ["table-wrap"]


def describe_table(tbl):
    """把表格转成一句占位描述（保留标签+说明文字），而不是把表格数字原样拼进正文"""
    label_node = tbl.find("label")
    caption_node = tbl.find("caption")
    label = label_node.text.strip() if label_node is not None and label_node.text else ""
    caption_text = "".join(caption_node.itertext()).strip() if caption_node is not None else ""
    desc = "：".join(x for x in [label, caption_text] if x)
    return f"[表格{('：' + desc) if desc else ''}，具体数据见原文表格]"


def extract_sections(filepath):
    """返回[{title, text}, ...]，已剔除公式源码（tex-math/MathML），表格替换为占位描述"""
    tree = parse(filepath, parser=xml_parser)
    root = tree.getroot()
    body = root.find(".//body")
    if body is None:
        return []
    strip_elements(body, *FORMULA_TAGS, with_tail=True)
    sections = []
    for sec in body.findall("sec"):  # 只取顶层章节，避免嵌套子章节重复计入
        title_node = sec.find("title")
        title = title_node.text if title_node is not None else None

        # 表格（<table-wrap>）里的<p>会被下面的findall(".//p")当成普通段落抓取，
        # 导致表格数字被拍扁拼接成不可读的文本流（如"0 (42.9)50 (35.7)..."）。
        # 先取出表格的标签+说明文字生成占位描述，再把整个table-wrap从树里剔除，
        # 避免它内部的<p>污染正文，同时保留"这里曾经有个表格"的语境供检索匹配。
        table_notes = [describe_table(tbl) for tbl in sec.findall(".//table-wrap")]
        strip_elements(sec, *TABLE_TAGS, with_tail=True)

        paragraphs = ["".join(p.itertext()).strip() for p in sec.findall(".//p")]
        paragraphs = [p for p in paragraphs if p]
        paragraphs.extend(table_notes)
        text = "\n\n".join(paragraphs) if paragraphs else "".join(sec.itertext()).strip()
        if text:
            sections.append({"title": title, "text": text})
    return sections


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=count_tokens,
)


def build_chunk(doc_id, chunk_index, text, source_title, section_title):
    """把一段文本包装成chunk字典，chunk_id = doc_id_c序号；total_chunks先留空，整篇处理完再回填"""
    return {
        "chunk_id": f"{doc_id}_c{chunk_index}",
        "text": text,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "total_chunks": None,  # 该文献处理完后统一回填
        "source_title": source_title,
        "section_title": section_title,
        "token_count": count_tokens(text),
    }


# ========== 3. 逐篇文献执行两层分割策略 ==========
print("\n===== 2. 执行文本分割策略 =====")

chunks = []
skipped_docs = []
docs_without_sections = 0

for i, row in df_raw.iterrows():
    doc_id = row["doc_id"]
    title = row.get("title") or ""
    filepath = os.path.join(DATA_DIR, row["filename"])

    try:
        sections = extract_sections(filepath)
    except Exception as e:
        skipped_docs.append({"doc_id": doc_id, "filename": row["filename"], "error": str(e)})
        sections = []

    doc_chunks = []

    if sections:
        for sec in sections:
            sec_text = sec["text"]
            sec_title = sec["title"]
            sec_tokens = count_tokens(sec_text)

            # 注意判断用的是TOKEN_LIMIT(512)不是CHUNK_SIZE(450)：章节只要不超过512就整段保留，
            # 超过512才用450的窗口去切，所以有的chunk会比450大一些，但不会超过嵌入模型上限
            if sec_tokens <= TOKEN_LIMIT:
                # b. 整体不分割：该章节本身在限制内，直接作为一个完整chunk
                doc_chunks.append(build_chunk(doc_id, len(doc_chunks), sec_text, title, sec_title))
            else:
                # a. 根据长度智能分割：章节超限，用滑动窗口二次切分
                for t in splitter.split_text(sec_text):
                    doc_chunks.append(build_chunk(doc_id, len(doc_chunks), t, title, sec_title))
    else:
        # 没有<sec>结构（或XML解析失败退化为空）：在全文层面做同样的判断
        docs_without_sections += 1
        full_text = row.get("body_text") or ""
        if full_text.strip():
            full_tokens = count_tokens(full_text)
            if full_tokens <= TOKEN_LIMIT:
                # b. 整体不分割：chunk_id直接用doc_id（跟任务要求的示例保持一致）
                doc_chunks.append({
                    "chunk_id": doc_id,
                    "text": full_text,
                    "doc_id": doc_id,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "source_title": title,
                    "section_title": None,
                    "token_count": full_tokens,
                })
            else:
                for t in splitter.split_text(full_text):
                    doc_chunks.append(build_chunk(doc_id, len(doc_chunks), t, title, None))

    total = len(doc_chunks)
    for c in doc_chunks:
        c["total_chunks"] = total
    chunks.extend(doc_chunks)

    if (i + 1) % 500 == 0:
        print(f"已处理 {i + 1}/{len(df_raw)}")

chunks_df = pd.DataFrame(chunks)
print(f"\n共生成 {len(chunks_df)} 个文本块，来自 {chunks_df['doc_id'].nunique()} 篇文献")
print(f"其中 {docs_without_sections} 篇没有<sec>结构，走的是全文层面的判断")
if skipped_docs:
    print(f"⚠️ {len(skipped_docs)} 篇文献XML解析失败被跳过（详见skipped_documents.csv）")

# 过滤低价值短文本块：抽查确认这类块基本是"None."(声明套话)、补充材料引用、
# 被截断的统计量残片等无信息量内容，入库前剔除，不占向量库空间。
# 剔除的块不会真正丢失，单独存一份dropped_short_chunks.csv方便回查。
MIN_CHUNK_CHARS = 20
before_filter = len(chunks_df)

text_len = chunks_df["text"].str.strip().str.len()
is_short = text_len < MIN_CHUNK_CHARS

# 安全网：找出"全部chunk都被判定为短块"的文献，各保留其中最长的一个，
# 避免这篇文献因为短块过滤被整篇清零、从语料库里彻底消失变得完全不可检索。
all_short_by_doc = is_short.groupby(chunks_df["doc_id"]).transform(lambda s: s.all())
if all_short_by_doc.any():
    rescue_pool = chunks_df.loc[all_short_by_doc, ["doc_id"]].copy()
    rescue_pool["_len"] = text_len[all_short_by_doc]
    rescue_idx = rescue_pool.groupby("doc_id")["_len"].idxmax()
    is_short.loc[rescue_idx] = False
    print(f"⚠️ {rescue_idx.nunique()} 篇文献的全部chunk原本都低于{MIN_CHUNK_CHARS}字符阈值，"
          f"已各保留其中最长的一个chunk兜底，避免整篇文献从语料库消失")

dropped_chunks_df = chunks_df[is_short].copy()
chunks_df = chunks_df[~is_short].reset_index(drop=True)
print(
    f"过滤低价值短文本块（<{MIN_CHUNK_CHARS}字符）: {before_filter} -> {len(chunks_df)}（剔除{len(dropped_chunks_df)}块）")

if len(dropped_chunks_df):
    dropped_chunks_df.to_csv(os.path.join(OUTPUT_DIR, "dropped_short_chunks.csv"), index=False)
    print("被剔除的短块已保存到 dropped_short_chunks.csv，供回查")

# 过滤后重新计算每篇文献的total_chunks（剔除短块后原来的计数会偏大）
chunks_df["total_chunks"] = chunks_df.groupby("doc_id")["doc_id"].transform("count")

# 最终校验：确认过滤没有让任何文献彻底消失（除非它本来就解析失败/无正文）
final_doc_count = chunks_df["doc_id"].nunique()
original_doc_count = df_raw["doc_id"].nunique()
vanished = original_doc_count - final_doc_count - len(skipped_docs)
if vanished > 0:
    print(f"⚠️ 仍有约 {vanished} 篇文献在最终数据集里没有任何chunk，且不属于已知的XML解析失败，建议再排查一下原因")
else:
    print("校验通过：除XML解析失败的文献外，没有其他文献因为过滤而彻底消失")

# ========== 4. 保存结果 + 处理统计信息 ==========
print("\n===== 3. 保存结果 =====")

output_path = os.path.join(OUTPUT_DIR, "chunks.parquet")
chunks_df.to_parquet(output_path, index=False)
chunks_df.to_json(os.path.join(OUTPUT_DIR, "chunks.jsonl"), orient="records", lines=True, force_ascii=False)

stats = {
    "processed_date": pd.Timestamp.now().isoformat(),
    "data_split": "full",
    "original_documents": len(df_raw),
    "skipped_documents": len(skipped_docs),
    "docs_without_section_structure": docs_without_sections,
    "dropped_short_chunks": len(dropped_chunks_df),
    "min_chunk_chars": MIN_CHUNK_CHARS,
    "total_chunks": len(chunks_df),
    "chunks_per_doc": len(chunks_df) / len(df_raw) if len(df_raw) > 0 else 0,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "token_limit": TOKEN_LIMIT,
    "output_file": str(output_path),
    "token_count_stats": chunks_df["token_count"].describe().to_dict(),
}
with open(os.path.join(OUTPUT_DIR, "processing_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

if skipped_docs:
    pd.DataFrame(skipped_docs).to_csv(os.path.join(OUTPUT_DIR, "skipped_documents.csv"), index=False)

print("处理配置和统计信息已保存到 processing_stats.json")
print(f"\n每篇文献平均块数: {stats['chunks_per_doc']:.2f}")
print("\nchunk token_count 分布:")
print(chunks_df["token_count"].describe())

# ========== 5. 预览结果 ==========
print("\n===== 4. 预览结果（前5个chunk） =====")
preview_cols = ["chunk_id", "doc_id", "chunk_index", "total_chunks", "section_title", "token_count"]
print(chunks_df[preview_cols].head(5).to_string())
print("\n第1个chunk的文本内容（前300字符）:")
print(chunks_df.iloc[0]["text"][:300])

# ========== 6. 质量验证 ==========
print("\n===== 5. 质量验证 =====")

# 6.1 有多少个块超过模型的token上限
over_limit = (chunks_df["token_count"] > TOKEN_LIMIT).sum()
print(f"超过{TOKEN_LIMIT} token上限的块数: {over_limit}/{len(chunks_df)} ({over_limit / len(chunks_df) * 100:.2f}%)")

# 6.2 文本质量：异常短的块
too_short = (chunks_df["text"].str.strip().str.len() < 20).sum()
print(f"文本长度异常短（<20字符）的块数: {too_short}")

# 6.3 是否包含标题（source_title缺失情况）
no_title = chunks_df["source_title"].isna().sum() + (chunks_df["source_title"].fillna("") == "").sum()
print(f"缺失来源标题(source_title)的块数: {no_title}")


# 6.4 文本是否被不完整截断（用结尾是否为常见标点做粗略代理指标）
def looks_truncated(text):
    """粗略判断文本是不是被截断了：结尾不是常见标点就返回True（只是估计，不一定准）"""
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.strip()
    return t[-1] not in ".!?\"'”）。！？：:"


sample_n = min(300, len(chunks_df))
quality_sample = chunks_df.sample(sample_n, random_state=42)
truncated_flag = quality_sample["text"].apply(looks_truncated)
print(f"抽样{sample_n}块中疑似被截断（结尾非常见标点，仅供参考）的块数: {truncated_flag.sum()}/{sample_n}")

# 6.5 对多块分割的文献重点检查：定位"同一章节被二次切分"产生的相邻chunk，验证重叠是否真的生效
# 注意：chunk_overlap只在RecursiveCharacterTextSplitter二次切分"同一个超长章节"时生效，
# 不同章节之间本来就不该有重叠——所以必须先筛出"同doc_id + 同section_title"的相邻chunk_index，
# 才是真正该检查重叠的对象，而不是随便挑相邻两块。
multi_chunk_doc_ids = chunks_df.loc[chunks_df["total_chunks"] > 1, "doc_id"].unique()
print(f"\n多块分割的文献数: {len(multi_chunk_doc_ids)}/{chunks_df['doc_id'].nunique()}")

print("\n--- 章节内滑动窗口重叠验证 ---")

split_pairs = []
grouped = chunks_df.groupby(["doc_id", "section_title"], dropna=False)
for (doc_id, sec_title), g in grouped:
    if len(g) < 2:
        continue
    g = g.sort_values("chunk_index")
    for j in range(len(g) - 1):
        split_pairs.append((doc_id, sec_title, g.iloc[j], g.iloc[j + 1]))

print(f"识别到 {len(split_pairs)} 对'同章节二次切分'产生的相邻chunk（重叠应该出现在这些对之间）")


def find_overlap(a_end: str, b_start: str, min_len: int = 15) -> str:
    """在a的结尾窗口里找b开头窗口的最长匹配前缀，粗略估计重叠内容"""
    max_check = min(len(a_end), len(b_start))
    for length in range(max_check, min_len, -1):
        if b_start[:length] in a_end:
            return b_start[:length]
    return ""


PREVIEW_CHARS = 400  # chunk_overlap=80 token通常远超80字符，窗口开小了根本看不到重叠内容

sample_pair_n = min(5, len(split_pairs))
if sample_pair_n:
    sample_idx = pd.Series(range(len(split_pairs))).sample(sample_pair_n, random_state=42)
    for idx in sample_idx:
        doc_id, sec_title, cur, nxt = split_pairs[idx]
        cur_end = cur["text"][-PREVIEW_CHARS:]
        nxt_start = nxt["text"][:PREVIEW_CHARS]
        print(f"\n--- 文献 {doc_id} / 章节《{sec_title}》 chunk{cur['chunk_index']} → chunk{nxt['chunk_index']} ---")
        print(f"  块{cur['chunk_index']}结尾: ...{cur_end}")
        print(f"  块{nxt['chunk_index']}开头: {nxt_start}...")
        overlap_guess = find_overlap(cur_end, nxt_start)
        if overlap_guess:
            print(f"  → 检测到疑似重叠片段（约{len(overlap_guess)}字符）: \"{overlap_guess}\"")
        else:
            print("  → 400字符窗口内没有直接匹配到重叠文本，可能是分割点恰好落在重叠边界外，建议手动核对该pair")
else:
    print(
        "没有找到'同章节二次切分'的相邻chunk样本（说明这批文献里没有章节超过512 token，不需要二次切分——可以对照章节级token分布结果确认是否合理）")

# 6.6 抽查异常短文本块（<20字符）具体内容，判断要不要在入库前过滤掉
print("\n--- 异常短文本块（<20字符）抽查 ---")
short_chunks = chunks_df[chunks_df["text"].str.strip().str.len() < 20]
print(f"共 {len(short_chunks)} 块（占比 {len(short_chunks) / len(chunks_df) * 100:.2f}%），抽样最多10条看具体内容:")
if len(short_chunks):
    preview = short_chunks[["chunk_id", "doc_id", "section_title", "text"]].head(10).copy()
    print(preview.to_string())

print("\n全部处理完成。")
print(f"chunk数据集: {output_path}")
print(f"处理统计报告: {os.path.join(OUTPUT_DIR, 'processing_stats.json')}")
