"""
检索流水线：把查询理解、多路检索、重排序串起来

检索系统开发第二部分的第3项。结合上周的10_query_understanding.py（查询理解），
加上这周新写的11_multi_path_retriever.py（向量+BM25多路检索、融合）和
12_reranker.py（多准则重排序），组成一条完整的检索流水线：

用户输入查询 -> 查询理解与增强 -> 多路检索+融合，得到一批候选 -> 重排序，得到最终参考文档列表

在整个流程里是第13步，检索部分到这里就完整了，后面的生成部分拿它的输出去用。
输入：先要有09_build_vector_index.py建好的chroma_db；运行后手动输入英文查询
输出：只是在屏幕上打印最终的参考文档列表（每条的各项分数、来源、完整chunk文本），不写文件
运行方式：python 13_retrieval_pipeline.py
    （启动时要加载embedding模型、构建BM25索引、加载reranker，比较慢；输入q退出）
    用importlib按数字开头的文件名导入10、11、12，所以要和它们放在同一个目录下运行
依赖：chromadb、sentence_transformers，以及10、11、12各自的依赖（rank_bm25、torch、transformers等）
"""

import importlib
import chromadb
from sentence_transformers import SentenceTransformer

# 文件名是数字开头，不能直接写import语句，所以用importlib按字符串名字导入
_m_query_understanding = importlib.import_module("10_query_understanding")
process_medical_query = _m_query_understanding.process_medical_query
CHROMA_DIR = _m_query_understanding.CHROMA_DIR
COLLECTION_NAME = _m_query_understanding.COLLECTION_NAME
EMBEDDING_MODEL_NAME = _m_query_understanding.EMBEDDING_MODEL_NAME
_m_multi_path_retriever = importlib.import_module("11_multi_path_retriever")
MultiPathRetriever = _m_multi_path_retriever.MultiPathRetriever
_m_reranker = importlib.import_module("12_reranker")
Reranker = _m_reranker.Reranker

# BM25索引先在一部分chunk上构建（避免第一次跑全量35万+条太慢/占内存太多，
# 先确认流程能跑通），确认没问题之后可以改成None，用全量数据构建
BM25_MAX_CHUNKS = 20000

FUSION_STRATEGY = "rrf"    # 'rrf' / 'weighted' / 'simple' / 'combmnz'，可以改这个值对比几种融合策略的效果
TOP_K_VECTOR = 15          # 向量检索候选数量
TOP_K_KEYWORD = 15         # BM25关键词检索候选数量
TOP_K_FUSED = 10           # 融合后送去重排序的候选数量
TOP_K_FINAL = 5            # 重排序后最终返回的参考文档数量


def print_final_results(results):
    """把重排序后的结果逐条打印出来：各项分数、来源信息和完整chunk文本。results是rerank()的返回列表"""
    if not results:
        print("没有筛选出符合条件的参考文档")
        return

    for i, item in enumerate(results):
        meta = item["metadata"]
        # token_count是08_chunk_documents.py当时用另一个分词器算的，跟reranker自己的
        # tokenizer不是同一个，只能大概参考，主要是看有没有明显超过512（reranker的
        # max_length），超过的话reranker实际只会看到chunk被截断之后的那一部分
        print(f"\n[{i + 1}] 最终分数={item['final_score']:.4f}"
              f"  相关性={item['relevance_score']:.4f}"
              f"  时效性={item['recency_score']:.4f}"
              f"  权威性={item['authority_score']:.4f}"
              f"  实体匹配={item['entity_match']}")
        print(f"来源={meta.get('source_title')}  期刊={item.get('journal')}  "
              f"年份={item.get('pub_year')}  章节={meta.get('section_title')}  "
              f"chunk_token数(粗略参考)={meta.get('token_count')}")
        # 之前这里只打印前300字，看不出reranker到底读到了什么内容，现在把全文打出来，
        # 方便对照上面的token数判断是不是被截断了、metformin到底有没有真的出现在chunk里
        print("--- 完整chunk文本 ---")
        print(item["text"])
        print("--- chunk文本结束 ---")


if __name__ == "__main__":
    print("加载embedding模型...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("连接向量索引...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print("索引中的向量数量:", collection.count())

    # 初始化多路检索器（会先构建一遍BM25索引，第一次跑这一步会比较慢，请耐心等）
    retriever = MultiPathRetriever(collection, embed_model, max_chunks=BM25_MAX_CHUNKS)

    # 加载重排序模型
    reranker = Reranker()

    print("\n完整检索流水线测试，输入q退出")
    while True:
        q = input("\n请输入查询: ")
        if q == "q":
            break

        # 第一步：查询理解与增强
        query_info = process_medical_query(q)
        print("清洗后:", query_info["cleaned_query"])
        print("识别实体:", query_info["entities"])

        # 第二步：多路检索 + 融合
        fused = retriever.retrieve(
            query_info,
            top_k_vector=TOP_K_VECTOR,
            top_k_keyword=TOP_K_KEYWORD,
            fusion_strategy=FUSION_STRATEGY,
            top_k=TOP_K_FUSED,
        )
        print(f"\n多路检索 + {FUSION_STRATEGY}融合后，候选数量: {len(fused)}")

        # 第三步：重排序，得到最终的参考文档列表
        # 现在传的是整个query_info（不只是查询文本），这样reranker才能拿到上周
        # 识别出来的实体词，做"文不对题"检查
        final_results = reranker.rerank(query_info, fused, top_k=TOP_K_FINAL)
        print("\n重排序后的最终参考文档列表:")
        print_final_results(final_results)
