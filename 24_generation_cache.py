"""
生成缓存 - GenerationCache 和 CachedPipeline

为什么要缓存：一次完整生成要调用4次本地大模型，实测每个问题60~100多秒。
同样的问题 + 同样的上下文再问一次，没必要重新生成。

设计（对应作业的四个要求）：
    1. 缓存键  = 对"查询 + 上下文 + 影响输出的设置"做SHA-256。输入完全一样，键就完全一样。
               设置包括：模型名、每个阶段的temperature/max_tokens、提示词模板内容的哈希、
               是否开了证据评估/批判性审查、上下文token预算。
               这样换了模型、改了提示词、改了温度之后，旧缓存不会被误用。
               查询只做空白符整理（多个空格压成一个），不转小写，因为医学缩写的大小写可能有意义。
    2. 大小限制  条数上限 max_entries，估算占用字节上限 max_bytes；超了就淘汰"最久没用过"的（LRU）。
               避免缓存越攒越多把内存占满。
    3. TTL     每条缓存有过期时间（默认6小时）。医学知识会更新，不能永远缓存。过期的条目取的时候发现就删掉。
    4. 温度限制  只缓存"低温度"的生成结果（流程里用到的所有阶段温度都不超过 max_cache_temperature，默认0.3）。
               温度高说明答案本来就带随机性，缓存一次的结果会让人以为每次都是这个答案，所以不缓存。
               注意：低温度只是"接近确定"，不是"完全确定"（Ollama这里没有固定seed），
               缓存下来的是第一次采样到的那个答案。

另外：失败的结果（草稿生成失败、答案为空）不缓存，否则一次偶然的失败会被保存下来反复返回。

CachedPipeline 把缓存套在 MedicalGenerationPipeline（20_medical_generation_pipeline.py）外面。
并发时同一个键只会生成一次：第二个线程会等第一个线程生成完，直接用缓存，不会白白再算一遍。

用法：
    cached = CachedPipeline(pipeline)                    # 不传cache就新建一个默认的GenerationCache
    result = cached.generate(query, retrieved_docs)      # 参数和 pipeline.generate 一样，多一个 use_cache
    result["cache"]                                      # {"hit": 是否命中, "stored": 是否存进缓存, "reason": 原因, ...}
    print(format_cache_stats(cached.stats()))            # 命中/未命中/淘汰/过期等统计

这个文件是模块，没有命令行参数，也不能直接运行。26_verify_eval_cache_batch.py（离线验证，假Ollama）和
27_test_evaluation_cache_batch.py（真实测试）会导入它；25_batch_generator.py 不导入它，
但批量处理时可以把 CachedPipeline 当 generator 传进去。
缓存放在内存里，进程结束就没了，不会写到磁盘。
"""

import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict

DEFAULT_MAX_ENTRIES = 100
DEFAULT_MAX_BYTES = 20 * 1024 * 1024        # 20MB（按JSON长度估算）
DEFAULT_TTL_SECONDS = 6 * 60 * 60           # 6小时
DEFAULT_MAX_CACHE_TEMPERATURE = 0.3


def normalize_query(query):
    """只整理空白符：首尾去掉，中间连续的空白压成一个空格"""
    return re.sub(r"\s+", " ", query).strip()


def make_cache_key(query, context_text, settings=None):
    """
    生成缓存键（64位十六进制字符串）。
    settings是影响输出的其他设置（字典）。用JSON序列化（key排序）再哈希，
    这样 查询/上下文/设置 之间不会因为拼接方式而混淆（比如 "ab"+"c" 和 "a"+"bc"）。
    """
    payload = {
        "query": normalize_query(query),
        "context": context_text,
        "settings": settings or {},
    }
    # sort_keys=True：字典里key的顺序不同也得到同一个键；default=str：设置里万一有JSON存不了的类型就转成字符串
    # 用SHA-256：不管输入多长键都是固定64位，输入差一个字键就完全不同，不用自己想办法拼字符串
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_size(result):
    """估算一个结果占多少字节：用JSON序列化后的UTF-8长度近似（不是精确的内存占用）"""
    return len(json.dumps(result, ensure_ascii=False, default=str).encode("utf-8"))


class GenerationCache:
    """
    放在内存里的生成结果缓存（线程安全）：有条数和字节上限，LRU淘汰，有TTL，只存低温度且成功的结果。
    用法：put(key, result, temperature) 存，get(key) 取，stats() 看统计。
    """

    def __init__(self, max_entries=DEFAULT_MAX_ENTRIES, max_bytes=DEFAULT_MAX_BYTES,
                 ttl_seconds=DEFAULT_TTL_SECONDS, max_temperature=DEFAULT_MAX_CACHE_TEMPERATURE,
                 time_fn=None):
        """
        Args:
            max_entries: 最多缓存多少条
            max_bytes: 估算的总字节数上限
            ttl_seconds: 过期时间（秒）。None或<=0表示不过期（不推荐）
            max_temperature: 超过这个温度的生成结果不缓存
            time_fn: 取当前时间的函数，默认time.time。测试的时候可以换成假时钟，不用真的等
        """
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.max_temperature = max_temperature
        self.time_fn = time_fn if time_fn is not None else time.time

        self._data = OrderedDict()       # key -> {"result", "created", "size"}；最近用过的放在最后
        self._total_bytes = 0
        # 批量处理是多线程同时存取，所以对_data、_total_bytes、_stats的读写都要拿这把锁
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0, "expirations": 0,
                       "skipped_high_temp": 0, "skipped_bad_result": 0, "skipped_too_large": 0}

    # ------------------------------------------------------------------
    def _is_expired(self, entry, now):
        """这条是不是已经过了TTL。ttl_seconds是None或<=0时永远不过期"""
        if not self.ttl_seconds or self.ttl_seconds <= 0:
            return False
        return now - entry["created"] >= self.ttl_seconds

    def _remove(self, key):
        """删掉一条，同时把它占的字节数从总数里减掉（调用的时候已经拿着锁）"""
        entry = self._data.pop(key)
        self._total_bytes -= entry["size"]

    def _purge_expired(self, now):
        # 调用的时候已经拿着锁
        expired = [k for k, e in self._data.items() if self._is_expired(e, now)]
        for k in expired:
            self._remove(k)
            self._stats["expirations"] += 1

    def _evict_if_needed(self):
        # 超过条数或字节上限，就从最久没用的开始删。调用的时候已经拿着锁
        while self._data and (len(self._data) > self.max_entries or self._total_bytes > self.max_bytes):
            # OrderedDict最前面的就是最久没用过的（get命中时会把用到的移到最后）
            oldest_key = next(iter(self._data))
            self._remove(oldest_key)
            self._stats["evictions"] += 1

    # ------------------------------------------------------------------
    def get(self, key, count_miss=True):
        """
        取缓存。count_miss=False时没命中不计入未命中次数（CachedPipeline里"拿到锁之后再查一次"用）。命中返回 (结果的深拷贝, 缓存信息)，没命中（或已过期）返回 (None, None)。
        返回拷贝是为了调用方改结果的时候不会改到缓存里的那份。
        """
        now = self.time_fn()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                if count_miss:
                    self._stats["misses"] += 1
                return None, None
            if self._is_expired(entry, now):
                self._remove(key)
                self._stats["expirations"] += 1
                if count_miss:
                    self._stats["misses"] += 1
                return None, None
            self._data.move_to_end(key)          # 刚用过，移到最后
            self._stats["hits"] += 1
            info = {"age_seconds": now - entry["created"], "size_bytes": entry["size"]}
            return copy.deepcopy(entry["result"]), info

    def put(self, key, result, temperature):
        """
        存缓存。返回 (是否存进去了, 原因)。
        temperature是这次生成用到的最高温度。不存的情况：温度太高、结果是失败的、单条就超过字节上限。
        """
        # 温度高（或者不知道温度）说明答案带随机性，缓存一次的结果会被当成"固定答案"反复返回，所以不存
        if temperature is None or temperature > self.max_temperature:
            with self._lock:
                self._stats["skipped_high_temp"] += 1
            return False, f"温度{temperature}超过上限{self.max_temperature}，不缓存"

        # 草稿阶段没成功（False），或者被跳过没有上下文（None），或者答案是空的，都当成失败，不存
        metrics = result.get("generation_metrics", {})
        draft_ok = metrics.get("stage_success", {}).get("answer_generation")
        if draft_ok is not True or not str(result.get("answer", "")).strip():
            with self._lock:
                self._stats["skipped_bad_result"] += 1
            return False, "生成失败或答案为空，不缓存"

        # 存的是深拷贝：调用方之后再改原来的result，不会改到缓存里的。拷贝和算大小是在拿锁之前做的
        stored = copy.deepcopy(result)
        size = estimate_size(stored)
        now = self.time_fn()
        with self._lock:
            # 单条就比总上限大的话，存进去只会把别的全挤掉然后自己也被淘汰，所以直接不存
            if size > self.max_bytes:
                self._stats["skipped_too_large"] += 1
                return False, f"这一条约{size}字节，超过总上限{self.max_bytes}，不缓存"
            # 同一个键再存：先删掉旧的，这样字节数不会重复累计，新的也会排到最后
            if key in self._data:
                self._remove(key)
            # 先清掉过期的，免得它们占着位置、把还有用的挤掉
            self._purge_expired(now)
            self._data[key] = {"result": stored, "created": now, "size": size}
            self._total_bytes += size
            self._stats["stores"] += 1
            self._evict_if_needed()
        return True, "已缓存"

    def clear(self):
        """清空所有缓存条目（统计数字不清零）"""
        with self._lock:
            self._data.clear()
            self._total_bytes = 0

    def __len__(self):
        with self._lock:
            return len(self._data)

    def __contains__(self, key):
        # 只看有没有（不算命中，也不更新使用顺序）；过期的算没有
        now = self.time_fn()
        with self._lock:
            entry = self._data.get(key)
            return entry is not None and not self._is_expired(entry, now)

    def stats(self):
        """返回统计字典的副本：各种计数，再加 entries（现在几条）、total_bytes、hit_rate（命中/(命中+未命中)）"""
        with self._lock:
            stats = dict(self._stats)
            stats["entries"] = len(self._data)
            stats["total_bytes"] = self._total_bytes
        lookups = stats["hits"] + stats["misses"]
        stats["hit_rate"] = (stats["hits"] / lookups) if lookups else 0.0
        return stats


# ---------------------------------------------------------------------------
# 把缓存套在流水线外面
# ---------------------------------------------------------------------------
class CachedPipeline:
    """
    给 MedicalGenerationPipeline 加上缓存的外壳：generate() 先查缓存，没命中才真的生成。
    同时有llm、assembler两个属性，可以直接当pipeline传给别的代码用。
    """

    def __init__(self, pipeline, cache=None):
        """
        Args:
            pipeline: MedicalGenerationPipeline
            cache: GenerationCache，不传就新建一个默认的
        """
        self.pipeline = pipeline
        self.cache = cache if cache is not None else GenerationCache()
        # 每个键一把锁：相同的问题互相等，不同的问题不用等，可以同时生成。
        # 用完的锁要从表里删掉（靠users计数），不然问题越多这个表越大
        self._inflight = {}                 # key -> {"lock": Lock, "users": 正在用这个锁的线程数}
        self._inflight_guard = threading.Lock()
        self._coalesced = 0                 # 因为等别的线程生成完而省下的生成次数

    # 这几个属性转发给里面的pipeline，这样CachedPipeline可以当pipeline用
    @property
    def llm(self):
        return self.pipeline.llm

    @property
    def assembler(self):
        return self.pipeline.assembler

    # ------------------------------------------------------------------
    def _stage_keys_used(self, enable_evaluation, enable_review):
        # 这次流程会用到哪几个阶段的模板
        keys = ["answer_generator"]
        if enable_evaluation:
            keys.append("evidence_evaluator")
        if enable_review:
            keys.extend(["critical_reviewer", "final_assembler"])
        return keys

    def effective_temperature(self, enable_evaluation, enable_review):
        """这次流程用到的所有阶段里最高的温度"""
        stages = self.pipeline.prompt_stages
        return max(stages[k].temperature for k in self._stage_keys_used(enable_evaluation, enable_review))

    def build_settings(self, enable_evaluation, enable_review, max_context_tokens):
        """影响输出的设置，进入缓存键。提示词模板用内容哈希，模板一改，旧缓存自动失效"""
        stages = self.pipeline.prompt_stages
        stage_info = {}
        for k in self._stage_keys_used(enable_evaluation, enable_review):
            stage = stages[k]
            template_text = stage.system_prompt + "\n##\n" + stage.user_prompt_template
            stage_info[k] = {
                "temperature": stage.temperature,
                "max_tokens": stage.max_tokens,
                "template_hash": hashlib.sha256(template_text.encode("utf-8")).hexdigest()[:16],
            }
        return {
            "model": getattr(self.pipeline.llm, "model_name", None),
            "stages": stage_info,
            "enable_evaluation": bool(enable_evaluation),
            "enable_review": bool(enable_review),
            "max_context_tokens": max_context_tokens,
        }

    # ------------------------------------------------------------------
    def _acquire_key_lock(self, key):
        """拿这个键对应的锁（没有就建一个）。users先加1再去抢锁，这样别的线程不会把还要用的锁删掉"""
        with self._inflight_guard:
            item = self._inflight.get(key)
            if item is None:
                item = {"lock": threading.Lock(), "users": 0}
                self._inflight[key] = item
            item["users"] += 1
        item["lock"].acquire()

    def _release_key_lock(self, key):
        """放掉这个键的锁；没有别的线程在用了，就把它从_inflight里删掉"""
        with self._inflight_guard:
            item = self._inflight[key]
            item["users"] -= 1
            if item["users"] == 0:
                del self._inflight[key]
        item["lock"].release()

    def _attach_cache_info(self, result, **info):
        """把缓存信息放进结果的"cache"字段，返回这个结果"""
        result["cache"] = info
        return result

    # ------------------------------------------------------------------
    def generate(self, query, retrieved_docs, enable_evaluation=None, enable_review=None,
                 max_context_tokens=None, use_cache=True):
        """
        和 MedicalGenerationPipeline.generate 用法一样，多了 use_cache（False就完全不用缓存）。
        结果里多一个 "cache" 字段：{"hit": 是否命中, ...}。
        流程：组装上下文 -> 算缓存键 -> 查缓存 -> 没命中就拿这个键的锁，再查一次，还没有才真的生成并尝试存入。
        """
        pipe = self.pipeline
        if enable_evaluation is None:
            enable_evaluation = pipe.enable_evaluation
        if enable_review is None:
            enable_review = pipe.enable_review
        if max_context_tokens is None:
            max_context_tokens = pipe.max_context_tokens

        if not use_cache:
            result = pipe.generate(query, retrieved_docs, enable_evaluation, enable_review, max_context_tokens)
            return self._attach_cache_info(result, hit=False, stored=False, reason="未启用缓存")

        # 先组装上下文（缓存键要用到），组装好的结果直接交给pipeline，不重复组装
        start = time.time()
        with pipe._assemble_lock:
            context_result = pipe.assembler.assemble(retrieved_docs, max_context_tokens=max_context_tokens)
        context_text = context_result["context_text"]

        def run_pipeline():
            return pipe.generate(query, retrieved_docs, enable_evaluation, enable_review,
                                 max_context_tokens, context_result=context_result)

        if not context_text.strip():
            # 没有上下文：pipeline会直接返回"没有证据"，不调用模型，没什么可缓存的
            return self._attach_cache_info(run_pipeline(), hit=False, stored=False, reason="上下文为空，不缓存")

        settings = self.build_settings(enable_evaluation, enable_review, max_context_tokens)
        key = make_cache_key(query, context_text, settings)

        cached, info = self.cache.get(key)
        if cached is not None:
            return self._finish_hit(cached, info, key, start, coalesced=False)

        # 没命中：同一个键同一时间只让一个线程生成，其他线程等着
        self._acquire_key_lock(key)
        try:
            # 等到锁之后再查一次：前面那个线程可能已经生成并存好了。
            # 这次没命中不再计入未命中（上面第一次查已经算过了）
            cached, info = self.cache.get(key, count_miss=False)
            if cached is not None:
                with self._inflight_guard:
                    self._coalesced += 1
                return self._finish_hit(cached, info, key, start, coalesced=True)

            # 温度取这次流程实际会用到的阶段里最高的，交给cache.put判断要不要存
            result = run_pipeline()
            temperature = self.effective_temperature(enable_evaluation, enable_review)
            stored, reason = self.cache.put(key, result, temperature)
            return self._attach_cache_info(result, hit=False, stored=stored, reason=reason,
                                           key=key[:12], temperature=temperature)
        finally:
            self._release_key_lock(key)

    def _finish_hit(self, cached, info, key, start, coalesced):
        """整理命中的结果：改写耗时、加上cache信息。coalesced=True表示是等别的线程生成完才拿到的"""
        # 命中：total_time_seconds改成这次实际花的时间，原来生成花的时间记在cache信息里
        metrics = cached["generation_metrics"]
        original_time = metrics["total_time_seconds"]
        metrics["total_time_seconds"] = time.time() - start
        return self._attach_cache_info(
            cached, hit=True, stored=False, reason="命中缓存", key=key[:12],
            original_time_seconds=original_time, age_seconds=info["age_seconds"], coalesced=coalesced)

    def stats(self):
        """缓存的统计，再加一项 coalesced（因为等别的线程而省下的生成次数）"""
        stats = self.cache.stats()
        with self._inflight_guard:
            stats["coalesced"] = self._coalesced
        return stats


def format_cache_stats(stats):
    """把stats()的结果整理成一行文字"""
    return (f"缓存：{stats['entries']}条，约{stats['total_bytes'] / 1024:.1f}KB；"
            f"命中{stats['hits']}次，未命中{stats['misses']}次（命中率{stats['hit_rate']:.0%}）；"
            f"存入{stats['stores']}，淘汰{stats['evictions']}，过期{stats['expirations']}，"
            f"因温度高没缓存{stats['skipped_high_temp']}，因结果失败没缓存{stats['skipped_bad_result']}"
            + (f"，并发合并{stats['coalesced']}次" if "coalesced" in stats else ""))
