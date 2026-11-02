"""第6步：按文章长度分层抽样，检查文章有没有清晰的章节结构。

做什么：
    把所有文章按 token 数分成短、中、长三档（三分位切），每档随机抽8篇（random_state=42），共24篇；
    重新读这24篇的XML，打印正文里每个<sec>的标题，看有多少篇有至少2个章节。
输入：
    pkl_model/parsed_articles_with_tokens.pkl（由 04_analyze_tokens.py 生成）
    medrag_data/pubmed_raw/extracted/oa_comm_aws/ 目录下抽中的XML文件
输出：
    pkl_model/stratified_samples.pkl（24篇抽样结果，多了 tier 列：短/中/长）
运行：
    python 06_stratified_sampling.py
    上一步 05_term_analysis.py，下一步 07_analyze_section_tokens.py
依赖：pandas、lxml
"""
import os
import pandas as pd
from lxml.etree import parse, XMLParser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_model")

df = pd.read_pickle(os.path.join(PKL_DIR, "parsed_articles_with_tokens.pkl"))

# ===== 按token数三等分，做分层抽样 =====
q33 = df['token_count'].quantile(1/3)
q66 = df['token_count'].quantile(2/3)
print(f"三分位切点: {q33:.0f} tokens / {q66:.0f} tokens")

short_df = df[df['token_count'] <= q33]
medium_df = df[(df['token_count'] > q33) & (df['token_count'] <= q66)]
long_df = df[df['token_count'] > q66]
print(f"短篇: {len(short_df)}篇 | 中篇: {len(medium_df)}篇 | 长篇: {len(long_df)}篇\n")

sample_short = short_df.sample(8, random_state=42).assign(tier="短")
sample_medium = medium_df.sample(8, random_state=42).assign(tier="中")
sample_long = long_df.sample(8, random_state=42).assign(tier="长")
samples = pd.concat([sample_short, sample_medium, sample_long])

print(samples[["filename", "tier", "token_count", "title"]].to_string())

# ===== 重新解析这24篇的章节结构（<sec><title>） =====
DATA_DIR = os.path.join(BASE_DIR, "medrag_data", "pubmed_raw", "extracted", "oa_comm_aws"
                                                                                      "")
parser = XMLParser(recover=True)

def extract_section_titles(filepath):
    """返回一篇文章正文里所有<sec>（包括嵌套的）的标题列表；没有body返回空列表，没有标题的写"(无标题)"。"""
    tree = parse(filepath, parser=parser)
    root = tree.getroot()
    body = root.find(".//body")
    if body is None:
        return []
    titles = []
    for sec in body.findall(".//sec"):
        title_node = sec.find("title")
        titles.append(title_node.text if title_node is not None else "(无标题)")
    return titles

print("\n===== 章节结构检查（抽样24篇） =====")
has_clear_structure = 0
for _, row in samples.iterrows():
    filepath = os.path.join(DATA_DIR, row['filename'])
    secs = extract_section_titles(filepath)
    print(f"{row['filename']} [{row['tier']}篇, {row['token_count']}tok]: {secs}")
    if len(secs) >= 2:  # 至少2个章节才算有清晰的结构
        has_clear_structure += 1

print(f"\n抽样中有清晰多章节结构的文章: {has_clear_structure}/24")

samples.to_pickle(os.path.join(PKL_DIR, "stratified_samples.pkl"))
print("已保存抽样结果")