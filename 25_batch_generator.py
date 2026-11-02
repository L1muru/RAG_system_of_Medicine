"""
批量生成 - batch_generate

用 ThreadPoolExecutor 并行处理多个问题。要求（对应作业）：
    1. 线程池并行：每个问题放进线程池里跑
    2. 线程数根据CPU核心数调整：不超过CPU核心数、不超过任务数，默认最多2个（见下面的说明）
    3. 单个任务失败不影响其他任务：每个任务里单独try/except，出错只记录在它自己的结果里
    4. 输入输出顺序一致：结果按任务的下标放进固定位置，不管谁先跑完

关于线程数，要说明一个限制：生成时绝大部分时间是在等本地Ollama返回，Ollama自己通常一次只处理
一个（或少数几个）请求，请求会在它那边排队。所以线程开多了，不一定更快，反而可能因为抢资源让单个请求更慢。
默认最多用2个线程，实际有没有加速要用真实运行的耗时来看（见 27_test_evaluation_cache_batch.py 的批量部分）。
这是我没法在没有Ollama的环境里验证的，只能在真实运行里确认。

每个任务可以是：
    (query, docs) 元组，或者
    {"query": ..., "docs": ..., "options": {"enable_review": False, ...}} 字典
generator可以是 MedicalGenerationPipeline（20_medical_generation_pipeline.py），也可以是
CachedPipeline（24_generation_cache.py，带缓存），只要有 generate(query, docs, **kwargs)。

用法：
    out = batch_generate(generator, tasks, max_workers=2, generate_kwargs={"enable_review": False})
    out["records"]    和tasks一一对应的结果列表（每条的字段见 batch_generate 的说明）
    out["summary"]    汇总：总数、成功/失败数、缓存命中数、线程数、总用时等
    print(format_batch_summary(out["summary"]))

这个文件是模块，没有命令行参数，也不能直接运行；26_verify_eval_cache_batch.py（离线验证）和
27_test_evaluation_cache_batch.py（真实测试）会调用它。
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MAX_WORKERS = 2


def choose_worker_count(num_tasks, requested=None, cpu_count=None, default_cap=DEFAULT_MAX_WORKERS):
    """
    决定线程数：
        没指定 requested：min(CPU核心数, default_cap, 任务数)
        指定了：min(requested, CPU核心数, 任务数)  （不让线程数超过CPU核心数，避免过度竞争）
        至少为1
    """
    # os.cpu_count()有可能返回None，这时按1个核心算
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    if requested is None:
        workers = min(cpu_count, default_cap)
    else:
        workers = min(requested, cpu_count)
    workers = min(workers, num_tasks)
    return max(1, workers)


def _parse_task(task):
    # 把任务统一成 (query, docs, options)；格式不对会抛ValueError，由调用处记成这个任务的失败
    if isinstance(task, dict):
        query = task.get("query")
        docs = task.get("docs")
        options = task.get("options") or {}
    elif isinstance(task, (tuple, list)) and len(task) == 2:
        query, docs = task
        options = {}
    else:
        raise ValueError(f"任务格式不对（需要 (query, docs) 或带query/docs的字典）：{type(task).__name__}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query为空或不是字符串")
    if not isinstance(docs, list):
        raise ValueError("docs不是列表")
    return query, docs, options


def _run_one(index, task, generator, generate_kwargs):
    # 在线程里执行一个任务。任何异常都被接住，变成这个任务自己的失败记录
    start = time.time()
    # 先把记录建好再开始try：不管中间哪一步出错，都能返回一条带index的记录，放回自己的位置
    record = {"index": index, "query": None, "success": False, "result": None, "error": None,
              "elapsed": 0.0, "cache_hit": False}
    try:
        query, docs, options = _parse_task(task)
        record["query"] = query
        # 先复制一份公共参数再用任务自己的options覆盖，不然一个任务的options会影响到别的任务
        kwargs = dict(generate_kwargs)
        kwargs.update(options)
        result = generator.generate(query, docs, **kwargs)
        record["result"] = result
        record["cache_hit"] = bool((result.get("cache") or {}).get("hit"))
        # pipeline遇到模型失败不会抛异常，而是在结果里标记；这里也算失败，但结果保留着
        if result["generation_metrics"]["stage_success"].get("answer_generation") is False:
            record["error"] = "答案生成阶段失败：" + str(result.get("answer"))
        else:
            record["success"] = True
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        # 任务格式不对时query还没解析出来，字典任务至少把原来的query记下来方便排查
        if record["query"] is None and isinstance(task, dict):
            record["query"] = str(task.get("query"))
    record["elapsed"] = time.time() - start
    return record


def batch_generate(generator, tasks, max_workers=None, generate_kwargs=None, progress=None):
    """
    并行批量生成。

    Args:
        generator: 有 generate(query, docs, **kwargs) 的对象
        tasks: 任务列表
        max_workers: 想用的线程数，实际数量由 choose_worker_count 决定
        generate_kwargs: 传给每次generate的公共参数，比如 {"enable_review": False}
        progress: 可选的回调 progress(已处理数, 总数, 这个任务的记录)，按任务顺序依次调用（前面的任务没跑完，后面的先跑完了也要等）
    Returns:
        {"records": [...], "summary": {...}}
        records和tasks一一对应、顺序相同；每条是
        {"index", "query", "success", "result", "error", "elapsed", "cache_hit"}
    """
    tasks = list(tasks)
    generate_kwargs = generate_kwargs or {}
    total = len(tasks)
    # 没有任务时也按1个任务算，这样summary里的线程数不会是0
    workers = choose_worker_count(max(total, 1), max_workers)

    start = time.time()
    records = [None] * total           # 按下标放结果，保证顺序和输入一致
    if total:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # 先全部提交，再按提交顺序一个个等结果。谁先跑完不重要，records[i]永远是tasks[i]的结果
            futures = [pool.submit(_run_one, i, task, generator, generate_kwargs) for i, task in enumerate(tasks)]
            done = 0
            for i, future in enumerate(futures):          # 按提交顺序取，所以records[i]对应tasks[i]
                try:
                    record = future.result()
                except Exception as e:                    # _run_one里已经接住了异常，这里是保险
                    record = {"index": i, "query": None, "success": False, "result": None,
                              "error": f"{type(e).__name__}: {e}", "elapsed": 0.0, "cache_hit": False}
                records[i] = record
                done += 1
                if progress is not None:
                    progress(done, total, record)
    wall = time.time() - start

    summary = {
        "total": total,
        "succeeded": sum(1 for r in records if r["success"]),
        "failed": sum(1 for r in records if not r["success"]),
        "cache_hits": sum(1 for r in records if r["cache_hit"]),
        "workers": workers,
        "cpu_count": os.cpu_count() or 1,
        "wall_seconds": wall,
        "sum_task_seconds": sum(r["elapsed"] for r in records),     # 所有任务耗时之和，≈串行跑要花的时间
    }
    return {"records": records, "summary": summary}


def format_batch_summary(summary):
    """把 batch_generate 返回的 summary 整理成一行文字"""
    return (f"批量：共{summary['total']}个，成功{summary['succeeded']}，失败{summary['failed']}，"
            f"缓存命中{summary['cache_hits']}；线程数={summary['workers']}（CPU核心数={summary['cpu_count']}）；"
            f"总用时{summary['wall_seconds']:.1f}秒，各任务耗时之和{summary['sum_task_seconds']:.1f}秒")
