"""第3步：检查 journal、pub_date、pmid/pmcid 这几个字段能不能用。

做什么：
    看期刊名有多少种、哪些最多；看发表年份是不是规整的4位数字、年份范围和每年篇数；
    看 pmid 和 pmcid 缺失情况，以及两个都缺（没法追溯原文）的有多少篇。
    只在屏幕上打印分析结果，不保存文件。
输入：
    pkl_model/parsed_articles_with_quality.pkl（由 02_check_quality.py 生成）
输出：
    无文件输出
运行：
    python 03_check_field_usability.py
    上一步 02_check_quality.py，下一步 04_analyze_tokens.py
依赖：pandas
"""
import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "pkl_model")

df = pd.read_pickle(os.path.join(PKL_DIR, "parsed_articles_with_quality.pkl"))

print("===== journal 字段分析 =====")
print(f"唯一期刊数: {df['journal'].nunique()}")
print(df['journal'].value_counts().head(15))

print("\n===== pub_date 字段分析 =====")
valid_year = df['pub_date'].apply(lambda x: bool(re.fullmatch(r"\d{4}", str(x))) if pd.notna(x) else False)
print(f"格式规整的4位年份: {valid_year.sum()} / {len(df)}")
if (~valid_year).sum() > 0:
    print("格式异常样例:")
    print(df[~valid_year]['pub_date'].value_counts().head(10))

# 转不成数字的年份变成NaN，这样才能求最小最大值和按年份计数
df['pub_year'] = pd.to_numeric(df['pub_date'], errors='coerce')
print(f"\n年份范围: {df['pub_year'].min():.0f} - {df['pub_year'].max():.0f}")
print(df['pub_year'].value_counts().sort_index())

print("\n===== pmid / pmcid 可追溯性分析 =====")
pmid_missing = df['pmid'].isna() | (df['pmid'].str.strip() == "")
pmcid_missing = df['pmcid'].isna() | (df['pmcid'].str.strip() == "")
both_missing = pmid_missing & pmcid_missing
print(f"pmid缺失: {pmid_missing.sum()} 篇")
print(f"pmcid缺失: {pmcid_missing.sum()} 篇")
print(f"pmid和pmcid同时缺失（完全无法追溯原文）: {both_missing.sum()} 篇")