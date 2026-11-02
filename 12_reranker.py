"""
重排序模块：Reranker

检索系统开发第二部分的第2项。11_multi_path_retriever.py融合出来的结果只是初步排序，
这里再用一个专门做相关性打分的reranker模型（BAAI/bge-reranker-base，一个cross-encoder，
把query和文档拼一起输入，直接给相关性打分，一般比向量检索的相似度更准）重新算一遍相关性，
然后再综合时效性、权威性这两个维度，一起决定最终排序。

时效性、权威性要用到文献的发表年份和期刊名，但这两个字段09_build_vector_index.py当时没有
存进chromadb的metadata里（METADATA_FIELDS里没有pub_date、journal）。好在最早解析XML的时候
01_parse_articles.py其实是提取过这两个字段的，存在pkl_model/parsed_articles.pkl里，所以这里
没有重新建索引，而是重新读一遍这个pkl文件，按doc_id对应起来，查表补上年份和期刊信息。

在整个流程里是第12步。这个文件只定义Reranker类，自己不能直接运行，由13_retrieval_pipeline.py
（以及16、21等脚本）用importlib导入后使用。
输入：10_query_understanding.process_medical_query()输出的query_info，
      和11_multi_path_retriever.py融合出来的候选chunk字典列表
输出：给每个候选补上relevance_score、recency_score、authority_score、final_score、
      entity_match、pub_year、journal这些字段，按final_score从高到低排好（可以只留前top_k个）
需要先有pkl_model/parsed_articles.pkl（01_parse_articles.py产出），第一次用会下载reranker模型
依赖：pandas、torch、transformers
"""

import os

import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARSED_ARTICLES_PATH = os.path.join(BASE_DIR, "pkl_model", "parsed_articles.pkl")

# 多准则重排序时各个维度的权重，相关性还是占大头
CRITERIA_WEIGHTS = {
    "relevance": 0.6,   # 相关性 - 基础要求
    "recency": 0.25,    # 时效性 - 提取年份，随年份线性衰减
    "authority": 0.15,  # 权威性 - 按照期刊添加不同权重
}

# 时效性打分：发表超过这么多年的文献，衰减到0分（线性衰减，自己拍的一个数字）
RECENCY_DECAY_YEARS = 10

# 权威性打分：几个知名医学期刊给更高的权重，写法跟上周的MEDICAL_SYNONYMS一样，
# 是自己手动列的一个简化版本，实际项目里应该用期刊影响因子(IF)数据库。
# 第一次实测发现命中率很低——真实语料库里出现的期刊（比如Frontiers in Immunology、
# Advances in Rheumatology这种专科/开放获取期刊）大多不在最初那份"综合性顶刊"清单里，
# 导致所有结果的权威性全都是默认值0.5，这一项完全没起到区分作用。这次多补了几个
# 心血管、内分泌、免疫风湿方向的专科期刊，但这仍然是手动列的一份不完整清单，覆盖不到的
# 期刊还是会退回默认值，想要更准确的话得接真正的期刊影响因子数据库。
JOURNAL_WEIGHTS = {
    # 综合性顶刊
    "the new england journal of medicine": 1.0,
    "new england journal of medicine": 1.0,
    "lancet": 1.0,
    "the lancet": 1.0,
    "jama": 0.95,
    "bmj": 0.9,
    "nature medicine": 0.9,
    "nature": 0.85,
    "science": 0.85,
    "nature communications": 0.75,
    "plos medicine": 0.7,
    "plos one": 0.55,
    # 心血管专科期刊
    "circulation": 0.8,
    "circulation research": 0.75,
    "european heart journal": 0.8,
    "journal of the american college of cardiology": 0.8,
    "jacc": 0.8,
    "atherosclerosis": 0.65,
    # 内分泌/糖尿病专科期刊
    "diabetes care": 0.75,
    "diabetologia": 0.75,
    "the lancet diabetes & endocrinology": 0.85,
    "endocrinology, diabetes & metabolism": 0.6,
    # 免疫/风湿专科期刊
    "frontiers in immunology": 0.55,
    "annals of the rheumatic diseases": 0.7,
    "arthritis & rheumatology": 0.7,
    "advances in rheumatology": 0.55,
}
DEFAULT_JOURNAL_WEIGHT = 0.5

# 没有年份信息的时候，时效性给个中间分，不确定是新是旧，不直接给0分
DEFAULT_RECENCY_SCORE = 0.5

# 实体不匹配的惩罚系数：上周10_query_understanding.py识别出了查询里的实体词
# （比如"metformin"），但重排序模型有时候会给一些完全没提到这个实体、只是
# 主题词面上沾点边（比如同样出现"cardiovascular"这个词）的内容打很高的相关性分数。
# 实测发现过明显的例子：一段讲GLP-1受体激动剂和减重手术的内容，全文没提metformin一次，
# 却被打了满分1.0的相关性。加一层实体检查能直接压低这种"文不对题"的结果。
ENTITY_MISMATCH_PENALTY = 0.6


def make_doc_id(row):
    # 跟08_chunk_documents.py里的make_doc_id逻辑保持一致，不然doc_id对不上，查表就查不到
    # （区别是pmid和pmcid都没有的时候这里返回None，不生成uuid，那种文献后面直接跳过）
    pmid = row.get("pmid")
    if pd.notna(pmid) and str(pmid).strip():
        return f"pmid_{str(pmid).strip()}"
    pmcid = row.get("pmcid")
    if pd.notna(pmcid) and str(pmcid).strip():
        return f"pmcid_{str(pmcid).strip()}"
    return None


def load_doc_metadata_lookup(pkl_path=PARSED_ARTICLES_PATH):
    # 从parsed_articles.pkl里把每篇文献的期刊、年份取出来，按doc_id做成一个查找表
    # 注意：08_chunk_documents.py里如果doc_id重复过，会加"_dup0"这种后缀，这里没有处理
    # 重复的情况，遇到重复doc_id的chunk，权威性/时效性这两项会查不到，会用默认值兜底
    df = pd.read_pickle(pkl_path)
    df = df[["pmid", "pmcid", "journal", "pub_date"]].copy()
    df["doc_id"] = df.apply(make_doc_id, axis=1)

    lookup = {}
    for _, row in df.iterrows():
        doc_id = row["doc_id"]
        if doc_id is None:
            continue
        pub_year = None
        if pd.notna(row["pub_date"]):
            # pub_date取的是年份字符串，所以只看前4位当年份，转不成整数就当没有年份
            try:
                pub_year = int(str(row["pub_date"])[:4])
            except ValueError:
                pub_year = None
        journal = row["journal"] if pd.notna(row["journal"]) else None
        lookup[doc_id] = {"journal": journal, "pub_year": pub_year}

    return lookup


class Reranker:
    """多准则重排序器：bge-reranker相关性 + 发表年份时效性 + 期刊权威性，主入口是rerank()"""

    def __init__(self, model_name=RERANKER_MODEL_NAME, doc_metadata_lookup=None):
        print(f"加载重排序模型 {model_name} ...")
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # doc_id -> {journal, pub_year}的查找表，不传的话就自己读pkl文件加载一份
        if doc_metadata_lookup is None:
            doc_metadata_lookup = load_doc_metadata_lookup()
        self.doc_metadata_lookup = doc_metadata_lookup

    def score_relevance(self, query, candidates):
        # 相关性打分：query和每个候选文本拼成一对，一起喂给模型
        # tokenize -> 推理 -> 转换为概率分数
        pairs = [[query, c["text"]] for c in candidates]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            # 模型输出的logits没有固定范围，过sigmoid压到0~1，才好跟时效性、权威性这两个0~1的分数加权
            logits = self.model(**inputs).logits.view(-1)
            scores = torch.sigmoid(logits).tolist()
        return scores

    def score_recency(self, pub_year):
        # 时效性：今年发表的接近1分，超过RECENCY_DECAY_YEARS年的老文献衰减到0分，线性衰减
        if pub_year is None:
            return DEFAULT_RECENCY_SCORE
        current_year = datetime.now().year
        age = current_year - pub_year
        age = max(age, 0)
        score = 1.0 - age / RECENCY_DECAY_YEARS
        score = max(0.0, min(1.0, score))
        return score

    def score_entity_match(self, keyword_terms, text):
        # 检查候选chunk的文本里有没有真的出现query_info["keyword_terms"]里的词
        # （上周process_medical_query()识别出的实体词+扩展的同义词）。
        # 没识别出任何实体的话就没法判断，返回None表示"这条不适用这个检查"。
        if not keyword_terms:
            return None
        text_lower = text.lower()
        for term in keyword_terms:
            if term in text_lower:
                return True
        return False

    def score_authority(self, journal):
        # 权威性：按期刊名查JOURNAL_WEIGHTS。
        # 一开始用的是精确匹配（字典的.get()），结果发现实测数据里期刊名经常带后缀，
        # 比如"Advances in Rheumatology (London, England)"，精确匹配对不上词典里的
        # "advances in rheumatology"，查不到就退回默认值，等于白列了。改成子串匹配：
        # 只要词典里的期刊名是完整期刊名字符串里的一部分就算命中。
        # 但子串匹配又会有新问题：比如"lancet"这个短词本身也是"the lancet diabetes &
        # endocrinology"的子串，如果随便取第一个匹配到的，可能会错误匹配到"lancet"
        # 这个权重（1.0）而不是更精确的"the lancet diabetes & endocrinology"（0.85）。
        # 所以这里在所有匹配到的候选里，选名字最长（最具体）的那个。
        if journal is None:
            return DEFAULT_JOURNAL_WEIGHT
        journal_lower = journal.strip().lower()
        matches = [(name, weight) for name, weight in JOURNAL_WEIGHTS.items() if name in journal_lower]
        if not matches:
            return DEFAULT_JOURNAL_WEIGHT
        _, best_weight = max(matches, key=lambda pair: len(pair[0]))
        return best_weight

    def rerank(self, query_info, candidates, criteria_weights=CRITERIA_WEIGHTS, top_k=None):
        """
        使用模型进行多准则重排序：相关性(reranker模型打分) + 时效性(按年份) + 权威性(按期刊)，
        再用上周10_query_understanding.py识别出的实体词做一层"文不对题"检查。

        query_info: 跟11_multi_path_retriever.py的retrieve()一样，直接接
                    10_query_understanding.process_medical_query()的输出，
                    这样上周（查询理解）和这周（重排序）的产出才真正连起来用，
                    不是只在打印里看一眼entities就完了。
        candidates: 11_multi_path_retriever.py融合出来的候选字典列表（会直接在这些字典上加字段）
        criteria_weights: 相关性/时效性/权威性三项的权重，默认用CRITERIA_WEIGHTS
        top_k: 只保留前多少个，None就全部返回

        返回按final_score从高到低排序的候选列表；candidates为空就返回空列表。
        """
        if not candidates:
            return []

        query = query_info["cleaned_query"]
        keyword_terms = query_info.get("keyword_terms", [])

        relevance_scores = self.score_relevance(query, candidates)

        for i, cand in enumerate(candidates):
            doc_id = cand["metadata"].get("doc_id")
            doc_meta = self.doc_metadata_lookup.get(doc_id, {})

            cand["relevance_score"] = relevance_scores[i]
            cand["recency_score"] = self.score_recency(doc_meta.get("pub_year"))
            cand["authority_score"] = self.score_authority(doc_meta.get("journal"))
            # 顺便把年份、期刊也记录下来，方便后面展示或者排查问题
            cand["pub_year"] = doc_meta.get("pub_year")
            cand["journal"] = doc_meta.get("journal")

            cand["final_score"] = (
                criteria_weights["relevance"] * cand["relevance_score"]
                + criteria_weights["recency"] * cand["recency_score"]
                + criteria_weights["authority"] * cand["authority_score"]
            )

            # 实体匹配检查：识别出了实体词，但这条内容里一个都没提到，说明reranker
            # 这次可能是被主题词面上的重叠骗了，给最终分数打个折扣压低排名，
            # 不是直接删掉（有可能是同义词覆盖不到，不想太武断地排除掉）
            entity_match = self.score_entity_match(keyword_terms, cand["text"])
            cand["entity_match"] = entity_match
            if entity_match is False:
                cand["final_score"] *= ENTITY_MISMATCH_PENALTY

        candidates.sort(key=lambda c: c["final_score"], reverse=True)

        if top_k is not None:
            candidates = candidates[:top_k]

        return candidates
