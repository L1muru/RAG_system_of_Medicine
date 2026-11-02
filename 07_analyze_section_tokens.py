"""第7步：统计每个章节（<sec>）的token长度。

做什么：
    对所有文章，重新读XML，取正文里顶层章节的全部文字，用 NeuML/pubmedbert-base-embeddings
    的分词器算每个章节的token数，打印分布、分位数和超过512 token的章节数，
    用来判断按章节切块时，章节会不会太长。
输入：
    pkl_model/parsed_articles_with_tokens.pkl（由 04_analyze_tokens.py 生成，这里只用到 filename 列）
    medrag_data/pubmed_raw/extracted/oa_comm_aws/ 目录下的XML
输出：
    pkl_model/sections_with_tokens.pkl（列：filename, section_title, token_count）
运行：
    python 07_analyze_section_tokens.py
    上一步 06_stratified_sampling.py
依赖：pandas、lxml、transformers
注意：第一次运行要联网下载分词器；要把全部文章重新读一遍并分词，会比较慢。
"""
import os
import pandas as pd
from lxml.etree import parse, XMLParser
from transformers import AutoTokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "medrag_data", "pubmed_raw", "extracted", "oa_comm_aws")
df = pd.read_pickle(os.path.join(BASE_DIR, "pkl_model", "parsed_articles_with_tokens.pkl"))

print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")
parser = XMLParser(recover=True)

def get_text(node):
    """取节点下全部文字拼成一个字符串（不像01里那样去公式，也不分段）；节点为None返回None。"""
    if node is None:
        return None
    return "".join(node.itertext()).strip()

def extract_sections_with_text(filepath):
    """返回一篇文章正文里每个顶层章节的 (标题, 全文) 列表；没有body或章节没内容就不返回。"""
    tree = parse(filepath, parser=parser)
    root = tree.getroot()
    body = root.find(".//body")
    if body is None:
        return []
    results = []
    for sec in body.findall("sec"):  # 只取顶层章节，避免嵌套子章节重复计入
        title_node = sec.find("title")
        title_text = title_node.text if title_node is not None else "(无标题)"
        full_text = get_text(sec)
        if full_text:
            results.append((title_text, full_text))
    return results

records = []
print("提取全部文章的章节并计算token数...")
for i, fname in enumerate(df['filename']):
    filepath = os.path.join(DATA_DIR, fname)
    try:
        for title, text in extract_sections_with_text(filepath):
            tok_count = len(tokenizer.encode(text, truncation=False))
            records.append({"filename": fname, "section_title": title, "token_count": tok_count})
    except Exception as e:
        print(f"跳过 {fname}: {e}")
    if (i + 1) % 500 == 0:
        print(f"已处理 {i + 1}/{len(df)}")

sec_df = pd.DataFrame(records)
print(f"\n共提取 {len(sec_df)} 个章节（来自{df['filename'].nunique()}篇文章）")
print(sec_df['token_count'].describe())

for p in [50, 75, 90, 95, 99]:
    val = sec_df['token_count'].quantile(p / 100)
    print(f"{p}%分位数: {val:.0f} tokens")

over_512 = (sec_df['token_count'] > 512).sum()
print(f"\n超过512 token的章节数: {over_512} / {len(sec_df)} ({over_512/len(sec_df)*100:.1f}%)")

sec_df.to_pickle(os.path.join(BASE_DIR, "pkl_model", "sections_with_tokens.pkl"))
print("已保存")