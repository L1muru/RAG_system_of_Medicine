"""第1步：把下载好的论文XML解析成一张表。

做什么：
    逐个读取 oa_comm_aws 目录下的XML，提取 pmid、pmcid、标题、期刊名、发表年份、摘要、
    正文（公式的LaTeX/MathML源码会被去掉），汇总成一个DataFrame存成pkl。
输入：
    medrag_data/pubmed_raw/extracted/oa_comm_aws/ 目录下的XML（由 00_download_oa_comm.py 下载）
输出：
    pkl_model/parsed_articles.pkl
    （列：filename, pmid, pmcid, title, journal, pub_date, abstract, body_text）
运行：
    python 01_parse_articles.py
    上一步 00_download_oa_comm.py，下一步 02_check_quality.py
依赖：pandas、lxml
注意：这个脚本没有 main 函数，直接运行就会开始解析整个目录。
"""
import os
import re
import pandas as pd
from lxml.etree import parse, XMLParser, strip_elements

# 数据源目录和解析结果的缓存位置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "medrag_data", "pubmed_raw", "extracted", "oa_comm_aws")
PKL_DIR = os.path.join(BASE_DIR, "pkl_model")

OUTPUT_PATH = os.path.join(PKL_DIR, "parsed_articles.pkl")
# recover=True：遇到格式小问题不整篇报错，尽量解析
parser = XMLParser(recover=True)

def extract_pmcid_from_filename(filename):
    """从文件名里取出 PMC 开头的编号（如 PMC123.1.xml 取到 PMC123），匹配不上返回None。"""
    match = re.match(r"(PMC\d+)", filename)
    return match.group(1) if match else None

FORMULA_TAGS = ["tex-math", "{http://www.w3.org/1998/Math/MathML}math"]

def get_text(node):
    """取一个XML节点里的纯文本：有<p>就按段落拼接（段间空两行），没有就直接取全部文字。节点为None返回None。
    注意：会直接删掉传进来的节点里的公式标签（原地修改，不是拷贝）。"""
    if node is None:
        return None
    strip_elements(node, *FORMULA_TAGS, with_tail=True)  # 剔除公式的原始LaTeX/MathML源码
    paragraphs = [
        "".join(p.itertext()).strip()
        for p in node.findall(".//p")
    ]
    paragraphs = [p for p in paragraphs if p]
    if paragraphs:
        return "\n\n".join(paragraphs)
    return "".join(node.itertext()).strip()  # 兜底：没有<p>标签的情况（比如摘要有时结构简单）
def parse_article(filepath):
    """解析单个XML文件，返回包含各字段的字典（某个标签找不到时对应字段是None）。"""
    tree = parse(filepath, parser=parser)
    root = tree.getroot()

    # PMID：PubMed的全局文献编号
    pmid_node = root.find(".//article-id[@pub-id-type='pmid']")
    pmid = pmid_node.text if pmid_node is not None else None

    # PMCID：PMC自己的编号，用来拼原文链接
    pmcid = extract_pmcid_from_filename(os.path.basename(filepath))

    # 标题
    title = get_text(root.find(".//article-title"))

    # 期刊名
    journal = get_text(root.find(".//journal-title"))

    # 发表年份：优先取网络优先发表(epub)，没有再取纸质发表(ppub)，再没有取pub；
    # 这几种都没有的话，最后随便取第一个pub-date标签兜底
    pub_date = None
    for pub_type in ("epub", "ppub", "pub"):
        date_node = root.find(f".//pub-date[@pub-type='{pub_type}']")
        if date_node is not None:
            year = date_node.find("year")
            pub_date = year.text if year is not None else None
            break
    if pub_date is None:
        date_node = root.find(".//pub-date")
        if date_node is not None:
            year = date_node.find("year")
            pub_date = year.text if year is not None else None

    # 摘要
    abstract = get_text(root.find(".//abstract"))

    # 正文全文（body标签下所有段落）
    body_text = get_text(root.find(".//body"))

    return {
        "filename": os.path.basename(filepath),
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title,
        "journal": journal,
        "pub_date": pub_date,
        "abstract": abstract,
        "body_text": body_text,
    }

# 下面是主流程：遍历目录里所有文件，单篇解析失败只打印提示，不影响其他文章
records = []
files = sorted(os.listdir(DATA_DIR))
print(f"共发现 {len(files)} 个XML文件，开始解析...")

for i, fname in enumerate(files):
    filepath = os.path.join(DATA_DIR, fname)
    try:
        records.append(parse_article(filepath))
    except Exception as e:
        print(f"解析失败: {fname} - {e}")
    if (i + 1) % 500 == 0:
        print(f"已处理 {i + 1}/{len(files)}")

df = pd.DataFrame(records)
df.to_pickle(OUTPUT_PATH)

print(f"\n解析完成，共{len(df)}条记录，已保存到: {OUTPUT_PATH}")
print("\n前3行预览:")
print(df.head(3))
print("\n各字段数据类型:")
print(df.dtypes)