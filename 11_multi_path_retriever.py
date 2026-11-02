"""
多路检索模块：MultiPathRetriever

检索系统开发第二部分的第1项。上周（10_query_understanding.py）已经有了向量检索，
关键词那一路用的是chromadb的子串匹配（$contains），比较粗糙。这周把关键词检索换成
真正的BM25算法，向量检索和BM25检索各搜一批候选，再用不同的融合策略合并成一份结果。

BM25索引是自己拿collection里全部的chunk文本构建的，没有用现成的搜索引擎（比如
Elasticsearch），用的是rank_bm25这个库，需要先pip install rank-bm25。

融合策略一共写了四种：简单合并去重（simple）、RRF（rrf）、加权融合（weighted），
再加一个加分项CombMNZ（combmnz），是信息检索里跟RRF、加权融合并列的经典融合方法。

在整个流程里是第11步。这个文件只定义类，自己不能直接运行，由13_retrieval_pipeline.py
（以及16、21等脚本）用importlib导入后使用，也就是 python 13_retrieval_pipeline.py 时会用到它。
输入：已经打开的chromadb collection（09建好的）、embedding模型、
      10_query_understanding.process_medical_query()输出的query_info
输出：融合之后的候选chunk字典列表（chunk_id、text、metadata、各种分数、fused_score等）
依赖：rank_bm25（pip install rank-bm25）
"""

import re

from rank_bm25 import BM25Okapi

# 构建BM25索引时，一批从chromadb里取多少条chunk，分批取避免一次性请求太大
BM25_LOAD_BATCH_SIZE = 5000

# 融合时各自先拿多少候选进来，实际返回给用户的数量由retrieve()的参数决定
DEFAULT_TOP_K = 10

# RRF融合用的常数，沿用上周的默认值
RRF_K = 60

# 加权融合里向量检索和关键词检索各自的权重，向量权重更高一些
VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4


def simple_tokenize(text):
    # 简单分词：转小写，只保留字母和数字，按非字母数字的部分切开
    # 没有用专门的分词库（比如nltk），中文和一些特殊词处理不了，先写一个能跑的版本
    return re.findall(r"[a-z0-9]+", text.lower())


class MultiPathRetriever:
    """向量检索 + BM25关键词检索 + 多种融合策略的检索器，主入口是retrieve()"""

    def __init__(self, collection, embed_model, max_chunks=None):
        """
        初始化：连接向量索引（用外面传进来的collection和embed_model，不在这里重新加载模型），
        同时把collection里的全部chunk取出来构建一份BM25索引。

        max_chunks: 调试用，限制只取前多少条chunk来建BM25索引，构建全量索引（35万+条）
        比较费时间和内存，先用一个小数字测试代码能不能跑通，跑通了再改成None跑全量。
        """
        self.collection = collection
        self.embed_model = embed_model

        print("正在从索引里加载全部chunk，构建BM25关键词索引...")
        self.chunk_ids, self.chunk_texts, self.chunk_metas = self._load_all_chunks(max_chunks)
        tokenized_corpus = [simple_tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25索引构建完成，共{len(self.chunk_ids)}条chunk")

    def _load_all_chunks(self, max_chunks):
        # 分批把collection里所有chunk的id、正文、metadata取出来
        all_ids = []
        all_texts = []
        all_metas = []
        offset = 0
        while True:
            batch = self.collection.get(
                limit=BM25_LOAD_BATCH_SIZE,
                offset=offset,
                include=["documents", "metadatas"],
            )
            batch_ids = batch["ids"]
            if not batch_ids:
                break
            all_ids.extend(batch_ids)
            all_texts.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])
            offset += BM25_LOAD_BATCH_SIZE
            print(f"  已加载 {len(all_ids)} 条...")
            if max_chunks is not None and len(all_ids) >= max_chunks:
                all_ids = all_ids[:max_chunks]
                all_texts = all_texts[:max_chunks]
                all_metas = all_metas[:max_chunks]
                break
        return all_ids, all_texts, all_metas

    def search_vector(self, query_embedding, top_k):
        # 向量检索：chromadb需要把查询转换成embedding向量再去搜
        # query_embedding是已经算好的查询向量，返回top_k条带向量距离/相似度的结果
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        formatted = []
        for i in range(len(ids)):
            formatted.append({
                "chunk_id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                # 距离转成相似度（越大越相关），方便跟BM25分数放在一起处理
                "vector_score": 1 - dists[i],
                "vector_distance": dists[i],
            })
        return formatted

    def search_keyword_bm25(self, query_text, top_k):
        # 关键词检索（BM25）：分词 -> 算BM25分数 -> 取分数最高的top_k -> 生成格式化结果
        # 注意BM25只在构建索引时取进来的那批chunk里搜（max_chunks限制的话就不是全量）
        tokenized_query = simple_tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # 分数从高到低排，取前top_k个的下标
        ranked_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indexes = ranked_indexes[:top_k]

        formatted = []
        for idx in top_indexes:
            if scores[idx] <= 0:
                # BM25分数为0说明查询词一个都没匹配上，这种不算真的"检索到"，跳过
                continue
            formatted.append({
                "chunk_id": self.chunk_ids[idx],
                "text": self.chunk_texts[idx],
                "metadata": self.chunk_metas[idx],
                "keyword_score": float(scores[idx]),
            })
        return formatted

    def fuse_simple(self, vector_results, keyword_results, top_k):
        # 简单合并去重：向量结果排前面，关键词结果里没出现过的接在后面，
        # 基本方法，没有真正把两边的排名/分数综合起来算
        seen = set()
        merged = []
        for item in vector_results + keyword_results:
            if item["chunk_id"] in seen:
                continue
            seen.add(item["chunk_id"])
            merged.append(item)
        return merged[:top_k]

    def fuse_rrf(self, vector_results, keyword_results, top_k):
        # RRF（Reciprocal Rank Fusion）：只看排名不看具体分数大小，
        # 排名越靠前加的分越高，两边都出现的片段分数会叠加，更适合学术检索场景
        scores = {}
        info = {}

        for rank, item in enumerate(vector_results):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            info[chunk_id] = item

        for rank, item in enumerate(keyword_results):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            # 两路都搜到的，直接在向量那条记录上补上keyword_score，不重复放两条
            if chunk_id in info:
                info[chunk_id]["keyword_score"] = item["keyword_score"]
            else:
                info[chunk_id] = item

        fused = []
        for chunk_id, item in info.items():
            item["fused_score"] = scores[chunk_id]
            fused.append(item)

        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:top_k]

    def fuse_weighted(self, vector_results, keyword_results, top_k):
        # 加权融合：把两边的分数分别做min-max归一化到0~1之间，再按权重加起来，
        # 向量检索权重更高（VECTOR_WEIGHT=0.6），因为语义相关性一般更可靠一些
        vector_norm = self._normalize_scores(vector_results, "vector_score")
        keyword_norm = self._normalize_scores(keyword_results, "keyword_score")

        info = {}
        for item in vector_results:
            chunk_id = item["chunk_id"]
            info[chunk_id] = dict(item)
            info[chunk_id]["norm_vector_score"] = vector_norm[chunk_id]
            info[chunk_id]["norm_keyword_score"] = 0.0

        for item in keyword_results:
            chunk_id = item["chunk_id"]
            if chunk_id in info:
                info[chunk_id]["keyword_score"] = item["keyword_score"]
                info[chunk_id]["norm_keyword_score"] = keyword_norm[chunk_id]
            else:
                new_item = dict(item)
                new_item["norm_vector_score"] = 0.0
                new_item["norm_keyword_score"] = keyword_norm[chunk_id]
                info[chunk_id] = new_item

        fused = []
        for chunk_id, item in info.items():
            item["fused_score"] = (
                VECTOR_WEIGHT * item["norm_vector_score"]
                + KEYWORD_WEIGHT * item["norm_keyword_score"]
            )
            fused.append(item)

        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:top_k]

    def fuse_combmnz(self, vector_results, keyword_results, top_k):
        # 加分策略：CombMNZ融合。跟加权融合一样先把两边分数归一化到0~1，
        # 但不是加权平均，而是直接相加，再乘以这条结果被几路检索命中（命中路数越多，
        # 乘的倍数越大）。这样"两路都搜到"的结果会被进一步放大，比单纯加权更强调
        # "多路一致"这个信号，是信息检索里跟RRF、加权融合并列的一个经典融合方法。
        vector_norm = self._normalize_scores(vector_results, "vector_score")
        keyword_norm = self._normalize_scores(keyword_results, "keyword_score")

        info = {}
        hit_count = {}

        for item in vector_results:
            chunk_id = item["chunk_id"]
            info[chunk_id] = dict(item)
            info[chunk_id]["norm_vector_score"] = vector_norm[chunk_id]
            info[chunk_id]["norm_keyword_score"] = 0.0
            hit_count[chunk_id] = hit_count.get(chunk_id, 0) + 1

        for item in keyword_results:
            chunk_id = item["chunk_id"]
            if chunk_id in info:
                info[chunk_id]["keyword_score"] = item["keyword_score"]
                info[chunk_id]["norm_keyword_score"] = keyword_norm[chunk_id]
            else:
                new_item = dict(item)
                new_item["norm_vector_score"] = 0.0
                new_item["norm_keyword_score"] = keyword_norm[chunk_id]
                info[chunk_id] = new_item
            hit_count[chunk_id] = hit_count.get(chunk_id, 0) + 1

        fused = []
        for chunk_id, item in info.items():
            score_sum = item["norm_vector_score"] + item["norm_keyword_score"]
            # MNZ：乘以命中路数（1路命中乘1，两路都命中乘2）
            item["fused_score"] = score_sum * hit_count[chunk_id]
            fused.append(item)

        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused[:top_k]

    def _normalize_scores(self, results, score_field):
        # min-max归一化，把一组分数缩放到0~1，方便跟另一路的分数加权相加
        norm = {}
        if not results:
            return norm
        values = [item[score_field] for item in results]
        min_v = min(values)
        max_v = max(values)
        for item in results:
            chunk_id = item["chunk_id"]
            if max_v == min_v:
                # 所有分数都一样的话（比如只有一条结果），直接给1，避免除以0
                norm[chunk_id] = 1.0
            else:
                norm[chunk_id] = (item[score_field] - min_v) / (max_v - min_v)
        return norm

    def retrieve(self, query_info, top_k_vector=15, top_k_keyword=15, fusion_strategy="rrf", top_k=DEFAULT_TOP_K):
        """
        多路检索的主入口：向量检索 + BM25关键词检索，再按选定的策略融合。

        Args:
            query_info: 查询处理器（10_query_understanding.process_medical_query）输出的查询信息
            top_k_vector: 向量检索返回数量
            top_k_keyword: 关键词检索返回数量
            fusion_strategy: 融合策略 ('rrf', 'weighted', 'simple', 'combmnz')
            top_k: 融合之后最终返回的数量

        Returns:
            融合后的候选chunk字典列表，按fused_score从高到低（simple策略是向量结果在前的顺序）
        """
        # 向量那路用query_info["vector_query"]（带BGE指令前缀），BM25那路用cleaned_query（不带前缀）
        query_embedding = self.embed_model.encode(query_info["vector_query"], normalize_embeddings=True)

        vector_results = self.search_vector(query_embedding, top_k_vector)
        keyword_results = self.search_keyword_bm25(query_info["cleaned_query"], top_k_keyword)

        if fusion_strategy == "simple":
            fused = self.fuse_simple(vector_results, keyword_results, top_k)
        elif fusion_strategy == "weighted":
            fused = self.fuse_weighted(vector_results, keyword_results, top_k)
        elif fusion_strategy == "rrf":
            fused = self.fuse_rrf(vector_results, keyword_results, top_k)
        elif fusion_strategy == "combmnz":
            fused = self.fuse_combmnz(vector_results, keyword_results, top_k)
        else:
            raise ValueError(f"不认识的fusion_strategy: {fusion_strategy}，只支持rrf/weighted/simple/combmnz")

        return fused
