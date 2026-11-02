"""
查询理解与增强模块

输入：用户的自然语言医学查询
输出：处理过后的查询信息，包括识别出的医学实体、扩展的同义词、
      向量检索用的查询、关键词检索用的查询、过滤条件

这里的同义词表和实体正则都是自己手写的一个简单版本，覆盖的词不多，
实际项目里应该用UMLS、MeSH这些医学词库来做，这里先写一个能跑起来的版本。

后面又加了检索部分：不只是拿向量查询去搜一遍就完事，而是向量检索
和关键词检索各搜一批候选，再把两边的结果合并、按相关度过滤一遍、
控制一下别都挤在同一篇文献里，最后整理成一个结构化的参考文档列表，
方便后面接大模型做生成。

在整个流程里是第10步，检索部分的开头，要先跑完09_build_vector_index.py建好chroma_db。
13_retrieval_pipeline.py、16_verify_generation_prep.py等会用importlib导入这个文件里的
process_medical_query等函数和路径常量。
运行方式：python 10_query_understanding.py
    （会加载embedding模型、连接chroma_db，然后进入交互循环，手动输入英文查询看结果，输入q退出）
依赖：chromadb、numpy、sentence_transformers
"""

import os
import re
from datetime import datetime

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# 使用静态医学同义词词典
# 实际应用中，这个词典应该更全面
# 可以从UMLS、MeSH等医学标准术语库构建
MEDICAL_SYNONYMS = {
    "mi": ["myocardial infarction", "heart attack"],
    "myocardial infarction": ["mi", "heart attack"],
    "htn": ["hypertension", "high blood pressure"],
    "hypertension": ["htn", "high blood pressure"],
    "t2dm": ["type 2 diabetes"],
    "cva": ["stroke", "cerebrovascular accident"],
    "stroke": ["cva"],
    "copd": ["chronic obstructive pulmonary disease"],
    "cvd": ["cardiovascular disease"],
    "cardiovascular disease": ["cvd"],
    "chf": ["congestive heart failure", "heart failure"],
    "bp": ["blood pressure"],
    "bmi": ["body mass index"],
}

# 创建医学实体识别正则
MEDICAL_PATTERNS = {
    "drug": r"\b(aspirin|metformin|atorvastatin|warfarin|insulin|ibuprofen|lisinopril)\b",
    # 使用单词边界\b确保完整匹配
    # 避免匹配到"aspirin"中的"pirin"
    "condition": r"\b(diabetes|hypertension|myocardial infarction|heart attack|stroke|"
                 r"cancer|tuberculosis|pneumonia|asthma|heart failure)\b",
    "lab_test": r"\b(hba1c|ldl|hdl|troponin|creatinine|glucose|cholesterol|blood pressure)\b",
}

BGE_QUERY_INSTRUCTION = "Represent this question for searching relevant passages: "

# 这几个路径和09_build_vector_index.py里保持一致，因为要连的是同一个已经建好的索引
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "medrag_pubmed_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# 检索相关的几个参数，都是自己拍的，具体效果要在真实索引上测过才知道准不准
VECTOR_CANDIDATES = 15      # 向量检索先取多少条候选
KEYWORD_CANDIDATES = 15     # 关键词检索先取多少条候选
DISTANCE_THRESHOLD = 0.6    # 向量距离超过这个值就认为跟查询不太相关，过滤掉（粗略估的，没有在真实数据上调过）
MAX_PER_SOURCE = 2          # 同一篇文献最多保留几个片段，避免结果全挤在一篇文章里
TOP_K = 8                   # 最后返回的参考文档数量
RRF_K = 60                  # RRF融合用的常数，用的是论文里给的默认值


def clean_query(query):
    # 基础清洗，去掉多余的空格（首尾空格去掉，中间连续的空白合并成一个空格）
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    return query


def find_entities(query):
    # 识别医学实体
    entities = {}
    for entity_type, pattern in MEDICAL_PATTERNS.items():
        found = re.findall(pattern, query, re.IGNORECASE)
        if found:
            # 去重一下，防止同一个词出现多次
            unique_found = []
            for word in found:
                if word.lower() not in [w.lower() for w in unique_found]:
                    unique_found.append(word)
            entities[entity_type] = unique_found
    return entities


def expand_synonyms(query):
    # 扩展医学同义词
    query_lower = query.lower()
    expanded_words = []
    for word, synonym_list in MEDICAL_SYNONYMS.items():
        if re.search(r"\b" + word + r"\b", query_lower):
            for synonym in synonym_list:
                if synonym not in expanded_words:
                    expanded_words.append(synonym)
    return expanded_words


def extract_filters(query):
    # 从查询中提取过滤条件（时间范围等）
    filters = {}

    # 匹配"近N年"
    match = re.search(r"近(\d+)年", query)
    if match:
        years = int(match.group(1))
        filters["year_from"] = datetime.now().year - years

    # 匹配"last N years"
    match = re.search(r"last (\d+) years?", query, re.IGNORECASE)
    if match:
        years = int(match.group(1))
        filters["year_from"] = datetime.now().year - years

    # 匹配"since 2020"这种年份
    match = re.search(r"since (\d{4})", query, re.IGNORECASE)
    if match:
        filters["year_from"] = int(match.group(1))

    # 注意：现在向量索引里每个chunk的元数据还没有存发表年份这个字段，
    # 所以这里提取出来的year_from暂时没法真正拿去做过滤，
    # 要等以后把pub_date加到元数据里、重新建一次索引才能用上

    return filters


def collect_keyword_terms(entities, expanded_terms):
    # 把识别出来的实体词和扩展出来的同义词收集到一起，去重，
    # 这些词后面用来做关键词检索（比直接拿整句话去匹配要准一些，
    # 不然一句话里的"what"、"the"这种词也会被当成关键词去搜）
    terms = []
    for entity_type, words in entities.items():
        for word in words:
            word_lower = word.lower()
            if word_lower not in terms:
                terms.append(word_lower)
    for word in expanded_terms:
        word_lower = word.lower()
        if word_lower not in terms:
            terms.append(word_lower)
    return terms


def process_medical_query(query):
    """
    处理医学查询，返回增强的查询信息
    """
    # 基础清洗
    cleaned_query = clean_query(query)

    # 识别医学实体
    entities = find_entities(cleaned_query)

    # 同义词扩展
    expanded_terms = expand_synonyms(cleaned_query)

    # 准备向量检索查询：BGE模型的最佳实践：添加指令前缀
    # （这里的前缀是直接写在f-string里的，上面的BGE_QUERY_INSTRUCTION常量在这个文件里没有被用到）
    base_query = f"Represent this question for searching relevant passages: {cleaned_query}"
    vector_query = base_query

    # 准备关键词检索查询
    if expanded_terms:
        keyword_query = cleaned_query + " " + " ".join(expanded_terms)
    else:
        keyword_query = cleaned_query

    # 关键词检索实际用的词表（比keyword_query更干净，只保留实体词和同义词）
    keyword_terms = collect_keyword_terms(entities, expanded_terms)

    # 从查询中提取过滤条件
    filters = extract_filters(cleaned_query)

    # 生成不同的查询版本
    result = {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "entities": entities,
        "expanded_terms": expanded_terms,
        "vector_query": vector_query,
        "keyword_query": keyword_query,
        "keyword_terms": keyword_terms,
        "filters": filters,
    }
    return result


def cosine_distance(vec_a, vec_b):
    # 算两个向量的余弦距离（1 - 余弦相似度），值越小越相似。
    # 索引建的时候用的就是余弦相似度，这里跟它保持一致。
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    cos_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return 1 - cos_sim


def search_documents(query_embedding, n_results=VECTOR_CANDIDATES):
    # 拿算好的查询向量去已经建好的chroma_db索引里做向量检索，
    # 这是第一路检索：靠语义相似度找相关片段
    # 注意：这里用的collection是全局变量，只有直接运行这个文件时（__main__里）才会创建，
    # 被别的脚本import之后单独调用这个函数会找不到collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )
    return results


def search_by_keywords(keyword_terms, n_results=KEYWORD_CANDIDATES):
    # 第二路检索：直接在文本里找有没有出现这些关键词（实体词+同义词），
    # 用chromadb的where_document做子串匹配，多个词之间是"或"的关系。
    # 这一路补的是向量检索可能漏掉的、字面上明确提到了这些词的片段。
    # 顺便把每个片段自己的embedding也取出来，这样后面才能给这些关键词命中的
    # 结果也算一个向量距离，不然它们没经过任何相关度筛选就直接进最终列表了。
    if not keyword_terms:
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    if len(keyword_terms) == 1:
        where_document = {"$contains": keyword_terms[0]}
    else:
        where_document = {"$or": [{"$contains": term} for term in keyword_terms]}

    results = collection.get(
        where_document=where_document,
        limit=n_results,
        include=["documents", "metadatas", "embeddings"],
    )
    return results


def fuse_results(vector_results, keyword_results, query_embedding):
    # 把向量检索和关键词检索两路结果融合起来，用的是Reciprocal Rank Fusion（RRF）：
    # 每个片段在某一路结果里排名越靠前，加的分越高，两路的分加起来就是融合分数。
    # 一个片段如果两路都搜到了，说明"语义相关"和"字面相关"都占了，更可信一些，
    # 这也是"多维度证据"的意思：不是只信一种检索方式的结果。
    scores = {}
    info = {}

    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]
    vector_metas = vector_results["metadatas"][0]
    vector_dists = vector_results["distances"][0]

    for rank in range(len(vector_ids)):
        chunk_id = vector_ids[rank]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        info[chunk_id] = {
            "chunk_id": chunk_id,
            "text": vector_docs[rank],
            "metadata": vector_metas[rank],
            "vector_distance": vector_dists[rank],
            "keyword_match": False,
        }

    keyword_ids = keyword_results["ids"]
    keyword_docs = keyword_results["documents"]
    keyword_metas = keyword_results["metadatas"]
    keyword_embeddings = keyword_results.get("embeddings")

    for rank in range(len(keyword_ids)):
        chunk_id = keyword_ids[rank]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        if chunk_id in info:
            info[chunk_id]["keyword_match"] = True
        else:
            # 关键词单独搜到的片段之前没有距离，导致完全不受相关度阈值约束，
            # 现在拿它自己的embedding跟查询向量算一个距离出来，
            # 这样它才能跟向量检索的结果用同一套标准去筛选
            if keyword_embeddings is not None and len(keyword_embeddings) > rank:
                distance = cosine_distance(query_embedding, keyword_embeddings[rank])
            else:
                distance = None
            info[chunk_id] = {
                "chunk_id": chunk_id,
                "text": keyword_docs[rank],
                "metadata": keyword_metas[rank],
                "vector_distance": distance,
                "keyword_match": True,
            }

    fused_list = []
    for chunk_id, item in info.items():
        item["fused_score"] = scores[chunk_id]
        fused_list.append(item)

    return fused_list


def filter_and_diversify(fused_list, top_k=TOP_K):
    # 精准筛选：先按融合分数从高到低排，
    # 再把向量距离太大的（说明跟查询语义上不太像）过滤掉，
    # 最后控制一下同一篇文献最多出现几次，避免参考文档列表被一两篇文章占满
    fused_list.sort(key=lambda item: item["fused_score"], reverse=True)

    filtered = []
    for item in fused_list:
        # 现在向量检索和关键词检索的结果都有distance了（关键词那批是补算的），
        # 统一用这一个阈值筛。万一某条实在算不出距离（比如索引没存embedding），
        # distance是None，这种就先放过，不因为技术原因误杀
        if item["vector_distance"] is not None and item["vector_distance"] > DISTANCE_THRESHOLD:
            continue
        filtered.append(item)

    source_count = {}
    diversified = []
    for item in filtered:
        source = item["metadata"].get("source_title", "unknown")
        count = source_count.get(source, 0)
        if count >= MAX_PER_SOURCE:
            continue
        source_count[source] = count + 1
        diversified.append(item)

    return diversified[:top_k]


def retrieve_references(query_info):
    # 把上面几步串起来：向量检索 + 关键词检索 -> 融合 -> 筛选去重
    # 返回的就是可以直接交给后面生成模块使用的参考文档列表
    # 查询向量只算一次，向量检索和后面给关键词命中结果补距离都用同一份
    # （embed_model也是__main__里创建的全局变量，跟search_documents用的collection一样）
    query_embedding = embed_model.encode(query_info["vector_query"], normalize_embeddings=True)
    vector_results = search_documents(query_embedding)
    keyword_results = search_by_keywords(query_info["keyword_terms"])
    fused_list = fuse_results(vector_results, keyword_results, query_embedding)
    references = filter_and_diversify(fused_list)
    return references


def format_references(references):
    # 整理成一份结构化的参考文档列表，字段都是后面生成模块大概率用得上的：
    # 出处、章节、原文片段、这条结果靠不靠谱的几个依据（距离、是否命中关键词、融合分数）
    ref_list = []
    for rank, item in enumerate(references):
        meta = item["metadata"]
        ref_list.append({
            "rank": rank + 1,
            "chunk_id": item["chunk_id"],
            "source_title": meta.get("source_title"),
            "section_title": meta.get("section_title"),
            "text": item["text"],
            "vector_distance": item["vector_distance"],
            "keyword_match": item["keyword_match"],
            "fused_score": round(item["fused_score"], 5),
        })
    return ref_list


def print_references(ref_list):
    # 把format_references整理好的列表打印出来看，每条只显示前300个字符
    if not ref_list:
        print("没有筛选出符合条件的参考文档")
        return

    for ref in ref_list:
        distance_str = f"{ref['vector_distance']:.4f}" if ref["vector_distance"] is not None else "无"
        print(f"\n[{ref['rank']}] 融合分数={ref['fused_score']}  向量距离={distance_str}  关键词命中={ref['keyword_match']}")
        print(f"来源={ref['source_title']}  章节={ref['section_title']}")
        print(ref["text"][:300] + "...")


if __name__ == "__main__":
    # 加载embedding模型，连接已经建好的向量索引（只用做一次，不用每次查询都重新加载）
    print("加载embedding模型...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("连接向量索引...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print("索引中的向量数量:", collection.count())

    # 自己手动输入查询来测试效果，输入q退出
    print("\n医学查询理解与增强 + 检索测试，输入q退出")
    while True:
        q = input("\n请输入查询: ")
        if q == "q":
            break

        result = process_medical_query(q)
        print("清洗后:", result["cleaned_query"])
        print("识别实体:", result["entities"])
        print("同义扩展:", result["expanded_terms"])
        print("向量查询:", result["vector_query"])
        print("关键词查询:", result["keyword_query"])
        print("关键词检索词表:", result["keyword_terms"])
        print("过滤条件:", result["filters"])

        # 向量检索 + 关键词检索，融合筛选后得到参考文档列表
        print("\n检索到的参考文档列表:")
        references = retrieve_references(result)
        ref_list = format_references(references)
        print_references(ref_list)

    # 注意：如果输入中文查询（比如"二甲双胍对心血管疾病有何影响？"），现在识别不到
    # 实体，因为词典和正则都是英文写的，索引里的PubMed文献也都是英文，这个后面
    # 可能要加一步翻译才能处理中文问题
