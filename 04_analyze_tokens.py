"""第4步：统计文章的token长度，并粗略测算容量（磁盘、切块数量、向量化速度）。

做什么：
    1. 用 NeuML/pubmedbert-base-embeddings 的分词器，算每篇"标题+摘要+正文"的token数，
       打印分布、分位数、超过512 token的文章数；
    2. 看磁盘剩余空间，留30GB余量后算出可用预算；
    3. 统计原始XML的平均大小；
    4. 抽样200篇，用 chunk_size=450、chunk_overlap=80 的切分器真实切一遍，看平均每篇切几块；
    5. 用这些块测向量化速度（代码里写的是 device="mps"，也就是苹果芯片的Mac），
       再临时建一个Chroma库测每块占多少存储（测完会删掉临时目录）；
    6. 综合上面的数据，估算按磁盘和按时间最多能处理多少篇文章。
输入：
    pkl_model/parsed_articles_with_quality.pkl（由 02_check_quality.py 生成）
    medrag_data/pubmed_raw/extracted/oa_comm_aws/ 目录下的XML（只用来统计文件大小）
输出：
    pkl_model/parsed_articles_with_tokens.pkl（比输入多了 full_text 和 token_count 两列）
    运行中会临时建 chroma_test_tmp 目录，结束时删除
运行：
    python 04_analyze_tokens.py
    上一步 03_check_field_usability.py，下一步 05_term_analysis.py
依赖：pandas、transformers、langchain_text_splitters、sentence_transformers、chromadb
注意：第一次运行要联网下载分词器和嵌入模型；向量化测速部分写死用mps，不是Mac的话需要改device。
"""
import os
import pandas as pd
from transformers import AutoTokenizer
import time
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "medrag_data", "pubmed_raw", "extracted", "oa_comm_aws")
PKL_DIR = os.path.join(BASE_DIR, "pkl_model")

df = pd.read_pickle(os.path.join(PKL_DIR, "parsed_articles_with_quality.pkl"))

print("加载tokenizer中（首次运行需联网下载，几十MB左右）...")
tokenizer = AutoTokenizer.from_pretrained("NeuML/pubmedbert-base-embeddings")

# 拼接"标题+摘要+正文"作为一篇文章完整Document的候选内容
def build_full_text(row):
    """把一行文章的标题、摘要、正文按顺序用空行拼成一段文字；空的部分和内容为"xx"的摘要会被跳过。"""
    parts = []
    if pd.notna(row['title']) and row['title'].strip():
        parts.append(row['title'])
    if pd.notna(row['abstract']) and row['abstract'].strip() and row['abstract'].strip() != "xx":
        parts.append(row['abstract'])
    if pd.notna(row['body_text']) and row['body_text'].strip():
        parts.append(row['body_text'])
    return "\n\n".join(parts)

df['full_text'] = df.apply(build_full_text, axis=1)

print(f"计算每篇文章的token数（{len(df)}篇，可能要几分钟，耐心等）...")
# truncation=False：不截断，要看的就是文章原本到底有多长
df['token_count'] = df['full_text'].apply(lambda t: len(tokenizer.encode(t, truncation=False)))

print("\n===== Token长度分布 =====")
print(df['token_count'].describe())

print("\n===== 关键分位数 =====")
for p in [50, 75, 90, 95, 99]:
    val = df['token_count'].quantile(p / 100)
    print(f"{p}%分位数: {val:.0f} tokens")

over_512 = (df['token_count'] > 512).sum()
print(f"\n超过512 tokens（嵌入模型上限）的文章数: {over_512} / {len(df)} ({over_512/len(df)*100:.1f}%)")

df.to_pickle(os.path.join(PKL_DIR, "parsed_articles_with_tokens.pkl"))
print("\n已保存到 parsed_articles_with_tokens.pkl")
# ===== 容量测算：磁盘空间 =====
total, used, free = shutil.disk_usage(os.path.expanduser("~"))
print("\n===== 磁盘空间 =====")
print(f"总容量: {total/1e9:.1f} GB | 已用: {used/1e9:.1f} GB | 可用: {free/1e9:.1f} GB")

SAFETY_MARGIN_GB = 30
usable_budget_gb = free / 1e9 - SAFETY_MARGIN_GB
print(f"预留{SAFETY_MARGIN_GB}GB安全余量后，可用预算: {usable_budget_gb:.1f} GB")

# ===== 原始XML真实平均大小 =====
xml_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".xml")]
total_xml_bytes = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in xml_files)
avg_xml_kb = total_xml_bytes / len(xml_files) / 1024
print(f"\n===== 原始XML大小 =====")
print(f"当前样本{len(xml_files)}篇，平均每篇: {avg_xml_kb:.1f} KB")

# ===== 真实切分：用实际的分割器切一遍 =====
# 长度按token数算（不是按字符数），这样和嵌入模型的512上限对得上
splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=80,
    length_function=lambda t: len(tokenizer.encode(t, truncation=False)),
)

print("\n模拟真实切分（抽样200篇代表比例）...")
sample_df = df.sample(200, random_state=42)
total_chunks = 0
total_chunk_chars = 0
sample_chunks_for_embedding = []

for _, row in sample_df.iterrows():
    chunks = splitter.split_text(row['full_text'])
    total_chunks += len(chunks)
    total_chunk_chars += sum(len(c) for c in chunks)
    sample_chunks_for_embedding.extend(chunks[:3])  # 每篇只留前3块，后面拿来测向量化速度和存储

avg_chunks_per_article = total_chunks / len(sample_df)
avg_chunk_chars = total_chunk_chars / total_chunks
print(f"平均每篇文章切出: {avg_chunks_per_article:.1f} 个chunk")
print(f"平均每个chunk: {avg_chunk_chars:.0f} 字符")

# ===== 向量化速度实测（M4, MPS加速） =====
print("\n===== 向量化速度测试 =====")
embed_model = SentenceTransformer("NeuML/pubmedbert-base-embeddings", device="mps")
test_chunks = sample_chunks_for_embedding[:200]
start = time.time()
embeddings = embed_model.encode(test_chunks, batch_size=32, show_progress_bar=False)
elapsed = time.time() - start
speed = len(test_chunks) / elapsed
print(f"{len(test_chunks)}个chunk耗时{elapsed:.1f}秒，速度: {speed:.1f} chunk/秒")

# ===== Chroma实际存储占用 =====
print("\n===== Chroma存储占用测试 =====")
CHROMA_TEST_DIR = os.path.join(BASE_DIR, "chroma_test_tmp")
client = chromadb.PersistentClient(path=CHROMA_TEST_DIR)
collection = client.create_collection("capacity_test")
collection.add(
    documents=test_chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(test_chunks))]
)

def get_dir_size(path):
    """统计一个目录（包括子目录）下所有文件加起来的字节数。"""
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)

chroma_bytes = get_dir_size(CHROMA_TEST_DIR)
bytes_per_chunk = chroma_bytes / len(test_chunks)
print(f"存入{len(test_chunks)}个chunk，Chroma占用{chroma_bytes/1e6:.2f}MB，平均每chunk: {bytes_per_chunk/1024:.2f} KB")

shutil.rmtree(CHROMA_TEST_DIR)

# ===== 综合测算上限 =====
print("\n===== 容量测算结果 =====")
storage_per_article_kb = avg_xml_kb + (avg_chunks_per_article * bytes_per_chunk / 1024)
print(f"每篇文章总占用（原始XML+向量库）约: {storage_per_article_kb:.2f} KB")

# 可用预算(GB)换算成KB，除以每篇占用，就是磁盘最多能放多少篇
max_by_disk = (usable_budget_gb * 1024 * 1024) / storage_per_article_kb
print(f"按磁盘空间上限，最多可支持: {max_by_disk:.0f} 篇文章")

# 每篇的向量化耗时 = 每篇块数 / 每秒能处理的块数，再反推给定小时数内能处理多少篇
seconds_per_article = avg_chunks_per_article / speed
for hours in [1, 3, 6, 12]:
    max_by_time = (hours * 3600) / seconds_per_article
    print(f"若向量化控制在{hours}小时内，最多能处理: {max_by_time:.0f} 篇文章")