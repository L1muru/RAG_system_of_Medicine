"""第2步：检查解析结果的数据质量。

做什么：
    统计各字段的缺失率，判断摘要缺失是否超过1%（决定要不要做填充策略）；
    再检查摘要/正文是否有异常过短的，以及正文有没有编码问题（乱码）。
    结果只在屏幕上打印，同时给表加上 abstract_len、body_len、body_encoding_issue 三列后存回pkl。
输入：
    pkl_model/parsed_articles.pkl（由 01_parse_articles.py 生成）
输出：
    pkl_model/parsed_articles_with_quality.pkl
运行：
    python 02_check_quality.py
    上一步 01_parse_articles.py，下一步 03_check_field_usability.py
依赖：pandas
"""
import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_model")

df = pd.read_pickle(os.path.join(PKL_DIR, "parsed_articles.pkl"))
print(f"总文献数: {len(df)}\n")

# 空字符串也算缺失（有些标签存在但内容是空的，不能只看isna）
def is_missing(series):
    """返回布尔Series：值为空(NaN/None)或者只有空白字符的位置是True。"""
    return series.isna() | (series.str.strip() == "")

print("===== 字段缺失率 =====")
for col in ["pmid", "pmcid", "title", "journal", "pub_date", "abstract", "body_text"]:
    missing_count = is_missing(df[col]).sum()
    missing_rate = missing_count / len(df) * 100
    print(f"{col:12s} 缺失 {missing_count:4d} 篇，缺失率 {missing_rate:.2f}%")

# 重点判断abstract是否需要清洗策略
abstract_missing_mask = is_missing(df["abstract"])
abstract_missing_rate = abstract_missing_mask.sum() / len(df) * 100
print(f"\nabstract缺失率: {abstract_missing_rate:.2f}%")

if abstract_missing_rate > 1:
    print("超过1%阈值，需要制定清洗策略")
    missing_abstract_df = df[abstract_missing_mask]
    body_available = (~is_missing(missing_abstract_df["body_text"])).sum()
    print(f"其中 {body_available}/{len(missing_abstract_df)} 篇有body_text，可以考虑用正文开头几句填充")
else:
    print("未超过1%阈值，缺失记录可以直接丢弃，不需要复杂填充策略")

# ===== 基础质量分析 =====
print("\n===== 基础质量分析 =====")

df["abstract_len"] = df["abstract"].fillna("").str.len()
short_abstracts = df[(df["abstract_len"] > 0) & (df["abstract_len"] < 50)]
print(f"异常极短摘要（非空但少于50字符）: {len(short_abstracts)} 篇")
if len(short_abstracts) > 0:
    print(short_abstracts[["filename", "abstract"]].head(5).to_string())

df["body_len"] = df["body_text"].fillna("").str.len()
short_bodies = df[(df["body_len"] > 0) & (df["body_len"] < 200)]
print(f"\n异常极短正文（非空但少于200字符）: {len(short_bodies)} 篇")

def has_encoding_issue(text):
    """判断一段文字有没有编码问题：含替换字符，或控制字符超过3个就算有。非字符串/空串返回False。"""
    if not isinstance(text, str) or text == "":
        return False
    if "\ufffd" in text:  # Unicode替换字符，典型的编码错误标志
        return True
    # 这个范围不包含制表符、换行、回车，因为正常文本里本来就有这些
    control_chars = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text))
    return control_chars > 3

df["body_encoding_issue"] = df["body_text"].apply(has_encoding_issue)
issue_count = df["body_encoding_issue"].sum()
print(f"\n正文疑似编码问题: {issue_count} 篇")
if issue_count > 0:
    print(df[df["body_encoding_issue"]][["filename"]].head(5).to_string())

df.to_pickle(os.path.join(PKL_DIR, "parsed_articles_with_quality.pkl"))
print("\n已保存质量分析结果到 parsed_articles_with_quality.pkl")