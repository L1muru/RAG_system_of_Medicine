"""第0步（整个流程的第一个脚本）：下载PMC开放获取论文的XML全文。

做什么：
    用NCBI的eSearch接口按"入库日期"分段查出符合许可条件（CC0 / CC BY / CC BY-SA / CC BY-ND，
    排除禁运）的PMC ID，再从AWS S3上的 pmc-oa-opendata 把每篇的XML下载到本地，
    一直下到凑够 TARGET_COUNT 篇为止。
输入：
    不需要本地输入文件，需要联网（eutils.ncbi.nlm.nih.gov 和 pmc-oa-opendata.s3.amazonaws.com）。
    如果输出目录里已经有一部分XML，会算进已有数量，只补下载差的部分（可以断点续传）。
输出（都在 medrag_data/pubmed_raw/ 下面）：
    extracted/oa_comm_aws/PMC<数字>.<版本>.xml  下载下来的论文XML
    oa_comm_aws_filelist.csv                    文件清单（filename, pmcid, version, bytes）
    oa_comm_aws_failed.txt                      下载失败的记录（只有出现失败时才会写）
运行：
    python 00_download_oa_comm.py
    下一步是 python 01_parse_articles.py
依赖：requests（其余是标准库）
"""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "medrag_data", "pubmed_raw")
OUT_DIR = os.path.join(RAW_DIR, "extracted", "oa_comm_aws")
FILELIST_PATH = os.path.join(RAW_DIR, "oa_comm_aws_filelist.csv")
FAILED_PATH = os.path.join(RAW_DIR, "oa_comm_aws_failed.txt")

TARGET_COUNT = 12150
WORKERS = 24
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESEARCH_HARD_LIMIT = 9999  # eSearch 无 history 时有效窗口上限约 10000
REQUEST_SLEEP = 0.34  # 无 API key 时建议 ≤3 次/秒

# 原 oa_comm ≈ 允许商业再利用的 CC 许可
COMMERCIAL_TERM = (
    "(cc0_license[filter] OR cc_by_license[filter] OR "
    "cc_by-sa_license[filter] OR cc_by-nd_license[filter]) "
    "AND open_access[filter] NOT pmc embargo[filter]"
)

# 分段查询最早回溯到此日期（足够覆盖补齐 12150）
SEARCH_START_DATE = date(2020, 1, 1)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "MedRAG-download/1.0 (research; local)"})


def _fmt(d: date) -> str:
    """把日期转成eSearch要求的 YYYY/MM/DD 格式。"""
    return d.strftime("%Y/%m/%d")


def fetch_pmc_ids(
    term: str,
    retmax: int = ESEARCH_HARD_LIMIT,
    retstart: int = 0,
) -> tuple[list[str], int]:
    """对任意 term 做 eSearch，返回 (idlist, count)。"""
    params = {
        "db": "pmc",
        "term": term,
        "retmax": str(retmax),
        "retstart": str(retstart),
        "retmode": "json",
        "sort": "date",
    }
    time.sleep(REQUEST_SLEEP)
    r = SESSION.get(EUTILS, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["esearchresult"]
    ids = data.get("idlist", [])
    total = int(data.get("count", 0))
    return ids, total


def fetch_ids_in_date_range(start: date, end: date) -> list[str]:
    """
    拉取 [start, end] 入库日期内的全部 PMC ID。
    若该窗口 count > 9999，则对半拆分递归，绕过 10000 上限。
    """
    term = f"({COMMERCIAL_TERM}) AND {_fmt(start)}:{_fmt(end)}[pmcrdat]"
    _, total = fetch_pmc_ids(term, retmax=0)
    print(f"  日期窗 {_fmt(start)} ~ {_fmt(end)}: count={total}")

    if total == 0:
        return []

    if total <= ESEARCH_HARD_LIMIT:
        ids, _ = fetch_pmc_ids(term, retmax=total, retstart=0)
        return ids

    # 单日仍超限：只能取前 9999（pmcrdat 最小粒度为天）
    if start == end:
        print(f"  警告: 单日 {_fmt(start)} 有 {total} 篇，超过上限，仅取前 {ESEARCH_HARD_LIMIT} 篇")
        ids, _ = fetch_pmc_ids(term, retmax=ESEARCH_HARD_LIMIT, retstart=0)
        return ids

    mid = start + (end - start) // 2
    left = fetch_ids_in_date_range(start, mid)
    right = fetch_ids_in_date_range(mid + timedelta(days=1), end)
    seen: set[str] = set()
    merged: list[str] = []
    # 左右两半合并后去重，保持原来的顺序
    for i in left + right:
        if i not in seen:
            seen.add(i)
            merged.append(i)
    return merged


def iter_date_windows(newest: date, oldest: date, step_days: int = 7):
    """从新到旧生成日期窗口（闭区间）。"""
    end = newest
    while end >= oldest:
        start = max(oldest, end - timedelta(days=step_days - 1))
        yield start, end
        end = start - timedelta(days=1)


def collect_todo_ids(need: int, have_nums: set[str]) -> list[str]:
    """
    按日期分段收集未下载 ID，直到凑够 need。
    突破 eSearch 全局前 10000 条限制。
    """
    todo: list[str] = []
    todo_set: set[str] = set()
    newest = date.today()
    oldest = SEARCH_START_DATE

    print(
        f"按 pmcrdat 分段收集 ID：需要 {need} 个新 ID，"
        f"已有本地 {len(have_nums)} 篇，回溯至 {_fmt(oldest)}"
    )

    for start, end in iter_date_windows(newest, oldest, step_days=7):
        if len(todo) >= need:
            break
        try:
            ids = fetch_ids_in_date_range(start, end)
        except Exception as e:
            print(f"  查询失败 {_fmt(start)}~{_fmt(end)}: {e}，跳过该窗")
            continue

        added = 0
        for i in ids:
            if i in have_nums or i in todo_set:
                continue
            todo.append(i)
            todo_set.add(i)
            added += 1
            if len(todo) >= need:
                break
        print(f"  本窗新增 {added}，累计待下载 {len(todo)}/{need}")

    if len(todo) < need:
        print(
            f"警告: 分段查询后仍只凑到 {len(todo)}/{need} 个新 ID，"
            f"可把 SEARCH_START_DATE 再往前调"
        )
    return todo[:need]


def candidate_urls(pmc_num: str) -> list[tuple[str, str]]:
    """生成候选 (prefix, xml_url)，优先 .1。
    S3上同一篇文章可能有多个版本（.1 .2 .3），不知道是哪个版本，所以都列出来依次试。"""
    urls = []
    for ver in (1, 2, 3):
        prefix = f"PMC{pmc_num}.{ver}"
        urls.append((prefix, f"{S3_BASE}/{prefix}/{prefix}.xml"))
    return urls


def download_one(pmc_num: str) -> tuple[str, str, str]:
    """
    下载单篇 XML。
    返回 (status, pmc_num, detail)
      status: ok | skip | fail
    """
    # 任何一个版本本地已经有（且不是空文件）就直接跳过，这样重跑不会重复下载
    for prefix, _ in candidate_urls(pmc_num):
        out_path = os.path.join(OUT_DIR, f"{prefix}.xml")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return "skip", pmc_num, os.path.basename(out_path)

    last_err = "xml_not_found"
    for prefix, url in candidate_urls(pmc_num):
        out_path = os.path.join(OUT_DIR, f"{prefix}.xml")
        try:
            with SESSION.get(url, timeout=120, stream=True) as resp:
                if resp.status_code == 404:
                    last_err = "http_404"
                    continue
                if resp.status_code != 200:
                    last_err = f"http_{resp.status_code}"
                    continue
                # 先写到 .part 临时文件，写完再改名，避免中断后留下写了一半的xml
                tmp_path = out_path + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, out_path)
            return "ok", pmc_num, os.path.basename(out_path)
        except Exception as e:
            last_err = str(e)
            continue
    return "fail", pmc_num, last_err


def write_filelist() -> int:
    """把输出目录里的xml文件整理成csv清单（文件名、pmcid、版本、字节数），返回xml文件数。"""
    files = sorted(f for f in os.listdir(OUT_DIR) if f.endswith(".xml"))
    with open(FILELIST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "pmcid", "version", "bytes"])
        for name in files:
            path = os.path.join(OUT_DIR, name)
            stem = name[:-4]
            parts = stem.rsplit(".", 1)
            pmcid = parts[0]
            version = parts[1] if len(parts) == 2 else ""
            w.writerow([name, pmcid, version, os.path.getsize(path)])
    return len(files)


def count_existing_xml() -> tuple[int, set[str]]:
    """返回 (xml文件数, 已有PMC数字ID集合)。"""
    have_nums: set[str] = set()
    n_files = 0
    if not os.path.isdir(OUT_DIR):
        return 0, have_nums
    for f in os.listdir(OUT_DIR):
        if not (f.startswith("PMC") and f.endswith(".xml")):
            continue
        n_files += 1
        body = f[3:-4]
        num = body.split(".", 1)[0]
        have_nums.add(num)
    return n_files, have_nums


def main() -> None:
    """主流程：统计已有文件 -> 收集待下载ID -> 多线程下载 -> 写失败记录和文件清单。"""
    os.makedirs(OUT_DIR, exist_ok=True)

    already, have_nums = count_existing_xml()
    print(f"输出目录: {OUT_DIR}")
    print(f"已有 XML: {already} 篇（唯一 PMCID: {len(have_nums)}）；目标: {TARGET_COUNT} 篇")

    if already >= TARGET_COUNT:
        n = write_filelist()
        print(f"已达目标，跳过下载。filelist 共 {n} 条 -> {FILELIST_PATH}")
        return

    need = TARGET_COUNT - already
    # 多取一些，应对个别 404 / 版本缺失
    todo = collect_todo_ids(need=need + 400, have_nums=have_nums)
    if not todo:
        raise SystemExit("未能获取到新的 PMC ID，请检查网络或许可过滤条件")
    print(f"待下载: {len(todo)} 篇（workers={WORKERS}）")

    ok = skip = fail = 0
    failed_rows: list[str] = []
    t0 = time.time()

    # 多线程并发下载；凑够目标数量就取消还没开始的任务，提前结束
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(download_one, pmc): pmc for pmc in todo}
        done = 0
        for fut in as_completed(futures):
            status, pmc_num, detail = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failed_rows.append(f"{pmc_num}\t{detail}")

            if done % 100 == 0 or already + ok >= TARGET_COUNT:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"进度 {done}/{len(todo)} | 库内约 {already + ok}/{TARGET_COUNT} | "
                    f"ok={ok} skip={skip} fail={fail} | {rate:.1f} 篇/秒"
                )
            if already + ok >= TARGET_COUNT:
                for f in futures:
                    f.cancel()
                break

    if failed_rows:
        with open(FAILED_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_rows) + "\n")
        print(f"失败记录: {FAILED_PATH} ({len(failed_rows)} 条)")

    n = write_filelist()
    elapsed = time.time() - t0
    print("\n===== 完成 =====")
    print(f"输出目录 XML 数: {n}")
    print(f"filelist: {FILELIST_PATH}")
    print(f"本次 ok={ok}, skip={skip}, fail={fail}, 耗时 {elapsed/60:.1f} 分钟")
    if n < TARGET_COUNT:
        print(f"警告: 未达目标 {TARGET_COUNT}，当前仅 {n}，可重新运行本脚本续传")


if __name__ == "__main__":
    main()
