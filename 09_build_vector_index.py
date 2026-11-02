"""
构建向量索引脚本（整个流程的第09步）

做什么：把08分好的文本块用嵌入模型（BAAI/bge-base-en-v1.5）转成向量，
写进ChromaDB本地持久化向量库，后面的检索都是在这个库上做的。
输入：chunks_output/chunks.parquet（08_chunk_documents.py产出）
产出：
    chroma_db/ 目录：ChromaDB向量库，合集名medrag_pubmed_chunks，用余弦距离，
        每个chunk的metadata存doc_id、chunk_index、total_chunks、source_title、
        section_title、token_count（注意没有存发表年份和期刊，12_reranker.py是另外去读pkl补的）
    chunks_output/vector_index_stats.json：这次建索引的统计信息
大致流程：加载嵌入模型（自动选mps/cuda/cpu）-> 读chunks.parquet -> 打开或新建合集
    （全量模式下合集已存在就断点续跑）-> 分批生成向量并写入（失败会重试）
    -> 校验向量数量 -> 跑一些质量验证（自相似性、空查询/超长查询、元数据过滤、示例检索）
SAMPLE_LIMIT设成数字就只处理随机抽样的那么多个chunk（小样本测试），设成None是跑全量。

运行方式：python 09_build_vector_index.py
    （不需要命令行参数；全量很慢，见下面SAMPLE_LIMIT旁边的注释，建议先小样本跑通）
下一步：10_query_understanding.py、11_multi_path_retriever.py等检索脚本都会连这个chroma_db
依赖：numpy、pandas、chromadb、sentence_transformers、torch（选设备时用），读parquet还需要pyarrow之类的引擎
"""

import os
import json
import time

import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ===== 路径与参数配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks_output")
CHUNKS_PATH = os.path.join(CHUNKS_DIR, "chunks.parquet")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
STATS_PATH = os.path.join(CHUNKS_DIR, "vector_index_stats.json")

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "medrag_pubmed_chunks"
BATCH_SIZE = 64  # 内存不足时调小这个值，比如32或16
# BGE官方建议：查询端加检索指令前缀能明显提升效果，文档端不用加
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# 全量355,695个chunk用bge-base实测预计约3小时（参考：之前capacity测算里pubmedbert-base在
# M4上是30 chunk/秒，bge-base同量级模型速度接近）。首次跑强烈建议先用小样本跑通全流程，
# 确认没问题后把SAMPLE_LIMIT改成None再挂着跑全量。
SAMPLE_LIMIT = None  # 500样本已验证通过，现在跑全量

def get_device() -> str:
    """按mps(苹果芯片) -> cuda -> cpu的顺序选能用的计算设备，返回设备名字符串"""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def clean_metadata_value(v):
    """Chroma的metadata只接受str/int/float/bool，把None、numpy类型统一转成原生Python类型"""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v

# ========== 1. 嵌入模型选择与加载 ==========
print(f"===== 1. 加载嵌入模型: {EMBEDDING_MODEL_NAME} =====")
device = get_device()
print(f"使用设备: {device}")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
embedding_dim = embed_model.get_embedding_dimension()
print(f"模型加载完成，向量维度: {embedding_dim}")

def embed_documents(texts, batch_size: int = BATCH_SIZE):
    """批量生成文档向量：BGE文档端不需要加指令前缀"""
    return embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # 归一化后配合余弦相似度索引，检索更快更稳
    )

def embed_query(query_text: str):
    """生成查询向量：BGE官方建议查询端加检索指令前缀"""
    return embed_model.encode(QUERY_INSTRUCTION + query_text, normalize_embeddings=True)

# ========== 2. 加载文本块数据集，环境准备 ==========
print("\n===== 加载文本块数据集 =====")
chunks_df = pd.read_parquet(CHUNKS_PATH)
before_na = len(chunks_df)
chunks_df = chunks_df[chunks_df["text"].notna() & (chunks_df["text"].str.strip() != "")].reset_index(drop=True)
if len(chunks_df) != before_na:
    print(f"⚠️ 剔除了 {before_na - len(chunks_df)} 条空文本记录")

if SAMPLE_LIMIT:
    chunks_df = chunks_df.sample(SAMPLE_LIMIT, random_state=42).reset_index(drop=True)
    print(f"⚠️ 当前是小样本测试模式，仅处理 {len(chunks_df)} 个chunk（SAMPLE_LIMIT={SAMPLE_LIMIT}）")
print(f"共加载 {len(chunks_df)} 个文本块")

# chunk_id在08_chunk_documents.py阶段已经生成且保证唯一（本质就是doc_id+章节内序号的组合），
# 直接复用作向量库的id，不用再重新拼一遍doc_id+chunk_index
assert chunks_df["chunk_id"].is_unique, "chunk_id存在重复，需要先检查chunks.parquet"

# ========== 3. 向量数据库配置与索引构建 ==========
print("\n===== 2. 构建ChromaDB持久化索引 =====")
os.makedirs(CHROMA_DIR, exist_ok=True)

import shutil
_free_gb = shutil.disk_usage(CHROMA_DIR).free / (1024 ** 3)
print(f"当前磁盘剩余空间: {_free_gb:.1f} GB")
if _free_gb < 5:
    print("⚠️ 剩余磁盘空间不足5GB，全量索引写入过程中可能因空间不足而失败，建议先清理磁盘")

try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
except Exception as e:
    # 比"某个合集读不出来"更严重的情况：连底层sqlite系统库都打不开（实测遇到过
    # "database disk image is malformed"），说明上次崩溃已经把数据库文件本身写坏了，
    # 不是能重试或续跑的状态。这种情况没法挽救已写入的向量，只能把整个chroma_db目录
    # 备份挪走、重新建一个干净的库，让脚本至少能自动往下跑，而不是卡死在这一步。
    print(f"⚠️ 打开ChromaDB失败，怀疑数据库文件已损坏: {e}")
    backup_dir = CHROMA_DIR + f".corrupted_{int(time.time())}"
    print(f"将损坏的chroma_db备份到 {backup_dir}，并重新创建一个空的索引目录")
    shutil.move(CHROMA_DIR, backup_dir)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

# 断点续跑逻辑：全量跑（SAMPLE_LIMIT is None）遇到已存在的合集时，不再无脑删除重建——
# 上一次全量运行如果是中途崩溃（比如跑到16万+个chunk时遇到过一次ChromaDB内部
# "Error in compaction...does not exist" 的瞬时故障），已经成功写入的部分不该被推倒重来，
# 否则每次崩溃都要从0开始，几个小时就白跑了。只有小样本测试模式（SAMPLE_LIMIT有值）才继续
# 沿用"先删除重建"的老逻辑，因为不同随机种子抽样容易产生id冲突脏数据。
n_total = len(chunks_df)
existing_names = [c.name for c in client.list_collections()]
pending_df = chunks_df
already_done = 0

resume_ok = False
if COLLECTION_NAME in existing_names and SAMPLE_LIMIT is None:
    # 上次正是在写入过程中崩溃的（ChromaDB内部compaction错误），已存在的合集本身有小概率
    # 处于同样受损的状态（get_collection/count/get都可能再次抛错）。这里做一次探测性尝试，
    # 任何一步失败就放弃续跑、老老实实删除重建，而不是让整个脚本在启动阶段就再崩一次。
    try:
        existing_collection = client.get_collection(COLLECTION_NAME)
        existing_count = existing_collection.count()
        if existing_count > 0:
            print(f"发现已存在的合集 {COLLECTION_NAME}，包含{existing_count}个向量，判定为断点续跑，不删除重建")
            existing_ids = set(existing_collection.get(include=[])["ids"])
            pending_df = chunks_df[~chunks_df["chunk_id"].isin(existing_ids)].reset_index(drop=True)
            already_done = n_total - len(pending_df)
            print(f"其中{already_done}个chunk已在索引中，跳过；本次剩余{len(pending_df)}个待写入")
            collection = existing_collection
            resume_ok = True
    except Exception as e:
        print(f"⚠️ 尝试续跑失败（已存在的合集可能处于损坏状态）: {e}")
        print("回退为删除重建（会丢失上次已写入但读不出来的部分，属已知代价）")

if not resume_ok:
    if COLLECTION_NAME in existing_names:
        print(f"发现已存在的合集 {COLLECTION_NAME}，先删除重建，避免id冲突导致脏数据")
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )

METADATA_FIELDS = ["doc_id", "chunk_index", "total_chunks", "source_title", "section_title", "token_count"]


def add_with_retry(ids, embeddings, documents, metadatas, max_retries=5, base_delay=5):
    """collection.add()偶发因ChromaDB内部compaction问题失败（曾实测跑到约16万个chunk时遇到过
    一次'Error in compaction: Error getting collection with segments...does not exist'），
    大概率是瞬时故障，做指数退避重试，避免一次偶发错误就让整个进程崩溃、前功尽弃。"""
    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            return
        except Exception as e:
            if attempt == max_retries:
                print(f"⚠️ 写入连续失败{max_retries}次，放弃重试，报错详情: {e}")
                raise
            print(f"⚠️ 写入失败（第{attempt}/{max_retries}次尝试）：{e}\n{delay}秒后重试...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


print(f"开始批量生成向量并写入索引（batch_size={BATCH_SIZE}，目标共{n_total}个chunk，"
      f"其中已完成{already_done}个，本次待处理{len(pending_df)}个）...")
t0 = time.time()
n_pending = len(pending_df)
for start in range(0, n_pending, BATCH_SIZE):
    # 只处理pending_df（续跑时已经在索引里的chunk已经被排除掉了）
    batch = pending_df.iloc[start:start + BATCH_SIZE]
    texts = batch["text"].tolist()
    embeddings = embed_documents(texts)
    metadatas = [
        {field: clean_metadata_value(row[field]) for field in METADATA_FIELDS}
        for _, row in batch.iterrows()
    ]
    add_with_retry(
        ids=batch["chunk_id"].tolist(),
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    # 每写够20个batch（或者最后一批）才打印一次进度，不然输出太多
    done_pending = min(start + BATCH_SIZE, n_pending)
    done_overall = already_done + done_pending
    if done_pending % (BATCH_SIZE * 20) == 0 or done_pending == n_pending:
        elapsed = time.time() - t0
        rate = done_pending / elapsed if elapsed > 0 else 0
        eta_min = (n_pending - done_pending) / rate / 60 if rate > 0 else float("inf")
        print(f"已写入 {done_overall}/{n_total}（本次运行速度{rate:.1f} chunk/秒，预计剩余 {eta_min:.1f} 分钟）")

elapsed_total = time.time() - t0
print(f"\n本次运行处理完成，耗时 {elapsed_total/60:.1f} 分钟")

# 验证索引大小
index_count = collection.count()
print(f"索引中实际向量数量: {index_count}（应等于目标chunk数 {n_total}）")
if index_count != n_total:
    print(f"⚠️ 索引数量和目标chunk数不一致，差 {abs(index_count - n_total)} 个，需要排查")
else:
    print("校验通过：索引向量数量与目标chunk数一致")

# 保存统计信息
stats = {
    "collection_name": COLLECTION_NAME,
    "total_chunks": index_count,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_dimension": embedding_dim,
    "index_built_at": pd.Timestamp.now().isoformat(),
    "chunk_size_stats": {
        "mean": float(chunks_df["token_count"].mean()),
        "max": float(chunks_df["token_count"].max()),
        "min": float(chunks_df["token_count"].min()),
    } if "token_count" in chunks_df.columns else {},
    "metadata_fields": METADATA_FIELDS,
    "build_seconds": elapsed_total,
    "device": device,
    "sample_limit": SAMPLE_LIMIT,
}
with open(STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
print(f"统计信息已保存到 {STATS_PATH}")

def query(query_text: str, n_results: int = 5, where_filter: dict | None = None):
    """用查询文本在索引里做向量检索，where_filter是可选的metadata过滤条件，返回chroma原始的查询结果字典"""
    query_embedding = embed_query(query_text)
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=where_filter,
    )

# ========== 4. 质量验证 ==========
print("\n===== 3. 质量验证 =====")

# 4.1 基础统计验证
print("\n--- 基础统计验证 ---")
print(f"向量数量: {collection.count()}")
peek = collection.peek(limit=3)
print("样本元数据抽查:")
for i, (cid, meta) in enumerate(zip(peek["ids"], peek["metadatas"])):
    print(f"  [{i}] id={cid} metadata={meta}")

# 4.2 相似性检索验证：拿索引里的chunk文本自己去查自己，预期应该以极高相似度排在检索结果第一位
print("\n--- 相似性检索验证（自相似性测试） ---")
self_test_sample = chunks_df.sample(min(5, len(chunks_df)), random_state=7)
self_hit = 0
for _, row in self_test_sample.iterrows():
    probe_text = row["text"][:200]  # 模拟"用户搜到了这段内容的一部分"
    result = query(probe_text, n_results=3)
    top_id = result["ids"][0][0] if result["ids"] and result["ids"][0] else None
    top_distance = result["distances"][0][0] if result["distances"] and result["distances"][0] else None
    is_self_hit = top_id == row["chunk_id"]
    self_hit += int(is_self_hit)
    dist_str = f"{top_distance:.4f}" if top_distance is not None else "N/A"
    print(f"查询chunk={row['chunk_id']} -> 检索到top1={top_id}（距离={dist_str}） "
          f"{'✓ 命中自己' if is_self_hit else '✗ 未命中自己，需要关注'}")
print(f"自相似性命中率: {self_hit}/{len(self_test_sample)}")

# 4.3 边界情况验证：空查询、超长查询
print("\n--- 边界情况验证 ---")
try:
    empty_result = query("", n_results=3)
    top1 = empty_result["ids"][0][0] if empty_result["ids"][0] else "无结果"
    print(f"空查询：正常返回，未报错（top1={top1}）")
except Exception as e:
    print(f"⚠️ 空查询报错: {e}")

try:
    long_query_text = "myocardial infarction risk factors " * 500  # 制造一个远超512 token的超长查询
    long_result = query(long_query_text, n_results=3)
    top1 = long_result["ids"][0][0] if long_result["ids"][0] else "无结果"
    print(f"超长查询：正常返回，未报错（应已被自动截断处理，top1={top1}）")
except Exception as e:
    print(f"⚠️ 超长查询报错: {e}")

# 4.4 元数据过滤功能验证
print("\n--- 元数据过滤功能验证 ---")
section_counts = chunks_df["section_title"].dropna()
section_counts = section_counts[section_counts != ""]
if len(section_counts):
    filter_value = section_counts.mode().iloc[0]
    filtered_result = query("treatment outcome", n_results=3, where_filter={"section_title": filter_value})
    n_hits = len(filtered_result["ids"][0]) if filtered_result["ids"] else 0
    print(f"按section_title='{filter_value}'过滤查询：返回{n_hits}条结果")
    if n_hits:
        hit_sections = {m.get("section_title") for m in filtered_result["metadatas"][0]}
        print(f"过滤是否生效（返回结果的section_title应全部等于过滤值）: {hit_sections}")
else:
    print("未找到可用于测试的section_title值，跳过该项")

# 4.5 真实医学查询示例，直观看检索效果
print("\n--- 示例检索：医学查询效果预览 ---")
demo_result = query("What are the risk factors for tuberculosis reinfection?", n_results=3)
for rank, (doc_id, doc_text, meta, dist) in enumerate(zip(
    demo_result["ids"][0], demo_result["documents"][0], demo_result["metadatas"][0], demo_result["distances"][0]
), start=1):
    title_preview = str(meta.get("source_title", ""))[:60]
    print(f"\n[{rank}] chunk_id={doc_id}  距离={dist:.4f}  来源标题={title_preview}")
    print(f"    {doc_text[:200]}...")

print("\n全部处理完成。")
print(f"向量库位置: {CHROMA_DIR}")
print(f"统计报告: {STATS_PATH}")
