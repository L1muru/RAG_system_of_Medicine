"""第5步：看看论文正文里的高频词和大写缩写情况。

做什么：
    统计正文里出现最多的30个单词（去掉停用词和长度不超过2的词），
    再统计全大写缩写（2到6个字母，如 DNA）出现在多少篇文章里、最常见的20个是什么。
    只在屏幕上打印，不保存文件。
输入：
    pkl_model/parsed_articles_with_tokens.pkl（由 04_analyze_tokens.py 生成，这里只用到 body_text 列）
输出：
    无文件输出
运行：
    python 05_term_analysis.py
    上一步 04_analyze_tokens.py，下一步 06_stratified_sampling.py
依赖：pandas（其余是标准库）
"""
import os
import re
from collections import Counter
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_pickle(os.path.join(BASE_DIR, "pkl_model", "parsed_articles_with_tokens.pkl"))

STOPWORDS = set("""the of and to in a is that for on with as by an be are this was at from or which have has it its
their they he she we you your our not but can may also into these those between using used use such other than
however results was were significant patients study data figure table shown observed""".split())

def tokenize_words(text):
    """转小写后只取连续的英文字母串，数字和符号都当作分隔。"""
    return re.findall(r"[A-Za-z]+", text.lower())

word_counter = Counter()
for text in df['body_text'].dropna():
    words = [w for w in tokenize_words(text) if w not in STOPWORDS and len(w) > 2]
    word_counter.update(words)

print("===== 高频词 Top 30（已去停用词） =====")
for word, count in word_counter.most_common(30):
    print(f"{word:20s} {count}")

def find_abbreviations(text):
    """找出文本里2到6个大写字母组成的独立单词，当作缩写（可能包含少数不是缩写的全大写词）。"""
    return re.findall(r"\b[A-Z]{2,6}\b", text)

abbrev_counter = Counter()
docs_with_abbrev = 0
for text in df['body_text'].dropna():
    abbrevs = find_abbreviations(text)
    if abbrevs:
        docs_with_abbrev += 1
    abbrev_counter.update(abbrevs)

print(f"\n===== 缩写使用情况 =====")
print(f"包含至少1个缩写的文章占比: {docs_with_abbrev}/{len(df)} ({docs_with_abbrev/len(df)*100:.1f}%)")
print("\n最高频的缩写 Top 20:")
for abbr, count in abbrev_counter.most_common(20):
    print(f"{abbr:10s} {count}")