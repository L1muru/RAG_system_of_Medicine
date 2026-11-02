"""
生成模块与提示词工程 第二部分 - 本地LLM集成（LLMGenerator）

通过Ollama的HTTP接口调用本地的生成模型（默认qwen3:8b）。这个类只负责"把提示词发给模型、
把模型的回复拿回来"，不关心提示词是什么内容，提示词怎么拼、四个阶段怎么串起来是
20_medical_generation_pipeline.py 的事。

主要功能：
    1. 初始化时测试能不能连上Ollama、模型有没有下载
    2. generate()：发送prompt（可以带system prompt），设置temperature和max_tokens；
       需要JSON输出时，会在system prompt里加上JSON格式要求，并且从模型的回复里把JSON
       提取出来、修补缺失的括号/引号（小模型输出的JSON经常被max_tokens截断，少个括号）
    3. generate_batch()：批量生成

直接运行这个文件可以做一个简单的自测（需要先启动Ollama）：
    python 19_llm_generator.py
没有命令行参数。自测会连本地Ollama依次做三个小测试（普通生成、要求JSON输出、批量生成），
把结果打印到屏幕，不写文件；连不上Ollama或者模型没下载就打印原因然后以状态码1退出。

输入输出：generate()的输入是prompt字符串（可选system prompt、temperature、max_tokens、json_format），
输出是一个结果字典（success、text、json、error、truncated、token数、耗时等，字段见generate()的说明）；
失败的时候不会抛异常，而是把原因放在字典的error里。
只用标准库（urllib、json、re），不需要装额外的包。
"""

import json
import re
import time
import urllib.error
import urllib.request

DEFAULT_MODEL_NAME = "qwen3:8b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 300           # 秒。8B模型在笔记本上生成上千个token要几十秒，留得宽一点
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1000

# 给Ollama设置的上下文窗口大小（num_ctx）。Ollama默认的窗口比较小，提示词太长会被悄悄截断，
# 所以显式设一个。上下文预算2000 + 提示词 + 上游阶段的输出，最坏情况加起来三千多token，8192够用
DEFAULT_NUM_CTX = 8192

CONNECTION_TEST_TIMEOUT = 5     # 测试连接时只等几秒，连不上就尽快报错

# 全角的逗号冒号，小模型输出JSON的时候偶尔会写成中文标点，修复JSON时换成英文的
FULLWIDTH_PUNCTUATION = {"，": ",", "：": ":"}


class LLMGenerator:
    """通过Ollama的HTTP接口调用本地模型的小封装：测试连接、拼消息、发请求、整理回复（去<think>、提取JSON）"""

    def __init__(self, model_name=DEFAULT_MODEL_NAME, base_url=DEFAULT_BASE_URL,
                 timeout=DEFAULT_TIMEOUT, default_temperature=DEFAULT_TEMPERATURE,
                 default_max_tokens=DEFAULT_MAX_TOKENS, num_ctx=DEFAULT_NUM_CTX,
                 merge_system_into_user=False, test_connection=True):
        """
        Args:
            model_name: Ollama模型名称
            base_url: Ollama服务地址
            timeout: 请求超时时间（秒）
            default_temperature: generate()没指定temperature时用的默认值
            default_max_tokens: generate()没指定max_tokens时用的默认值
            num_ctx: 上下文窗口大小
            merge_system_into_user: 有的模型不支持system角色，设成True就把system prompt
                                    拼到用户消息前面一起发（qwen3支持system角色，保持False）
            test_connection: 初始化时是否测试连接
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.num_ctx = num_ctx
        self.merge_system_into_user = merge_system_into_user

        # 不走系统代理：如果电脑上开了代理（环境变量里有HTTP_PROXY），urllib默认会把
        # 访问localhost的请求也发给代理，结果连不上本地的Ollama
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        self.available = False
        self.connection_message = "还没有测试连接"
        if test_connection:
            self.test_connection()

    # ------------------------------------------------------------------
    # 连接测试
    # ------------------------------------------------------------------
    def _request(self, method, path, payload=None, timeout=None):
        # 发一个HTTP请求，返回解析好的JSON（字典）。出错的时候直接抛异常，由调用的地方处理
        url = self.base_url + path
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with self._opener.open(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_connection(self):
        """测试能不能连上Ollama，以及要用的模型有没有下载。返回True/False，详细信息存在self.connection_message里"""
        try:
            data = self._request("GET", "/api/tags", timeout=CONNECTION_TEST_TIMEOUT)
        except Exception as e:
            self.available = False
            self.connection_message = (f"连不上Ollama（{self.base_url}）：{type(e).__name__}: {e}。"
                                       f"请先在另一个终端运行 ollama serve")
            return False

        installed = []
        for m in data.get("models", []):
            installed.append(m.get("name") or m.get("model") or "")

        found = False
        for name in installed:
            if name == self.model_name or name == self.model_name + ":latest":
                found = True
            # 写的模型名没带版本标签（比如"qwen3"）的时候，只要有同名的任何版本就算找到
            if ":" not in self.model_name and name.split(":")[0] == self.model_name:
                found = True

        if not found:
            self.available = False
            self.connection_message = (f"Ollama在运行，但是没有找到模型 {self.model_name}。"
                                       f"已有的模型：{installed}。请先运行 ollama pull {self.model_name}")
            return False

        self.available = True
        self.connection_message = f"已连接Ollama（{self.base_url}），模型 {self.model_name} 可用"
        return True

    # ------------------------------------------------------------------
    # 构建提示词
    # ------------------------------------------------------------------
    def _json_instruction(self, json_format):
        # 需要JSON输出时，加在system prompt末尾的格式要求
        if isinstance(json_format, (dict, list)):
            example = json.dumps(json_format, ensure_ascii=False)
        else:
            example = str(json_format)
        return ("\n\n【输出格式要求】\n"
                "只输出一个合法的JSON对象，不要输出JSON以外的任何文字，不要使用markdown代码块。"
                "JSON的结构如下：\n" + example)

    def _build_system_text(self, system_prompt, json_format):
        """system prompt（没有就当空串）；要JSON输出的话在末尾接上格式要求，最后去掉首尾空白"""
        text = system_prompt or ""
        if json_format:
            text = text + self._json_instruction(json_format)
        return text.strip()

    def _build_messages(self, prompt, system_prompt=None, json_format=None):
        # 根据模型的格式组装消息。用Ollama的/api/chat接口，消息分system和user两种角色，
        # Ollama会按模型自己的对话模板（qwen是ChatML格式）把它们拼成真正喂给模型的文本，
        # 所以这里不需要自己手写模板标记。
        # 如果换成不支持system角色的模型，把merge_system_into_user设成True，
        # 就把system prompt拼到用户消息前面
        system_text = self._build_system_text(system_prompt, json_format)
        if system_text and self.merge_system_into_user:
            return [{"role": "user", "content": system_text + "\n\n" + prompt}]
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": prompt})
        return messages

    def build_full_prompt(self, prompt, system_prompt=None, json_format=None):
        """把system和user拼成一段文字，用来看"最后发给模型的到底是什么"（打日志、估算token数）"""
        messages = self._build_messages(prompt, system_prompt, json_format)
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]\n{m['content']}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # 处理模型的回复
    # ------------------------------------------------------------------
    def _clean_text(self, text):
        # qwen3有"思考模式"，思考内容会包在<think>...</think>里，答案里不需要
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "<think>" in text:
            # 只有开头没有结尾：说明思考到一半就被max_tokens截断了，后面的都是思考内容
            text = text.split("<think>")[0]
        return text.strip()

    def extract_json(self, text):
        """
        从模型的回复里提取JSON。返回(解析出来的对象, 是否成功)。
        处理的情况：JSON前后夹着别的文字、被```json代码块包着、括号/引号被截断少了。
        """
        if not text:
            return None, False

        candidate = text
        # 模型有时会把JSON包在```json ... ```里（代码块可能没写结尾就被截断了，所以结尾允许是```或者文本末尾），
        # 里面确实以{或[开头才取代码块里的内容，否则还是用整段文字
        fence = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
        if fence and fence.group(1).strip()[:1] in ("{", "["):
            candidate = fence.group(1)

        # 从第一个{（没有就找[）开始取，把前面的废话去掉
        start = candidate.find("{")
        if start == -1:
            start = candidate.find("[")
        if start == -1:
            return None, False
        candidate = candidate[start:]

        # 第一次尝试：直接解析。raw_decode允许JSON后面还跟着别的文字
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
            return obj, True
        except json.JSONDecodeError:
            pass

        # 第二次尝试：修补之后再解析
        repaired = self.repair_json(candidate)
        if repaired is not None:
            try:
                return json.loads(repaired), True
            except json.JSONDecodeError:
                pass
        return None, False

    def _try_parse(self, text):
        # 把多余的逗号（逗号后面直接跟着}或]）去掉再解析，成功返回去掉多余逗号之后的文本，失败返回None
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            return None

    def repair_json(self, text):
        """
        修补不完整的JSON文本，补上缺失的引号和括号。修好了返回字符串，修不好返回None。

        做法：从头扫一遍，记录当前有哪些括号还没闭合，同时记下每个"字符串外面的逗号"的位置。
        1. 先试：在末尾补上没闭合的引号和括号（没写完的"key": 补一个null）
        2. 不行的话，退一步：从后往前，在某个逗号处截断（丢掉最后那个写了一半的元素），
           再补括号。丢掉的只是被截断的那一小部分
        """
        out = []                # 扫描过程中保留下来的字符
        stack = []              # 还没闭合的括号，里面存的是将来要补的闭合符号
        cut_points = []         # 每个逗号：(逗号之前已经保留了多少个字符, 那时的括号栈)
        in_string = False
        escape = False

        for ch in text:
            if in_string:
                out.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            # 字符串外面的全角逗号/冒号换成英文的（字符串里面的内容不动，上面已经continue了）
            ch = FULLWIDTH_PUNCTUATION.get(ch, ch)
            if ch == '"':
                in_string = True
                out.append(ch)
            elif ch == "{":
                stack.append("}")
                out.append(ch)
            elif ch == "[":
                stack.append("]")
                out.append(ch)
            elif ch in "}]":
                if ch not in stack:
                    continue            # 多出来的闭合符号，丢掉
                while stack[-1] != ch:  # 中间漏了闭合符号的话，先补上
                    out.append(stack.pop())
                stack.pop()
                out.append(ch)
                if not stack:
                    break               # 最外层已经闭合，后面的内容不要了
            elif ch == ",":
                cut_points.append((len(out), list(stack)))
                out.append(ch)
            else:
                out.append(ch)

        if in_string:
            if escape and out and out[-1] == "\\":
                out.pop()
            out.append('"')

        # 第一种修法：直接在末尾补
        tail = "".join(out).rstrip()
        if tail.endswith(":"):
            tail += " null"
        fixed = self._try_parse(tail + "".join(reversed(stack)))
        if fixed is not None:
            return fixed

        # 第二种修法：退到最近的逗号处截断
        for pos, saved_stack in reversed(cut_points):
            prefix = "".join(out[:pos])
            fixed = self._try_parse(prefix + "".join(reversed(saved_stack)))
            if fixed is not None:
                return fixed
        return None

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------
    def generate(self, prompt, system_prompt=None, temperature=None, max_tokens=None, json_format=None):
        """
        生成一次回复。

        Args:
            prompt: 用户消息（具体的问题、证据等）
            system_prompt: 系统提示（角色和规则），可以不传
            temperature: 温度，不传用默认值
            max_tokens: 最多生成多少个token，不传用默认值
            json_format: 需要JSON输出时，传JSON结构的示例（字典或字符串），
                         会加在system prompt末尾，并且从回复里提取JSON。不需要JSON就不传

        Returns:
            字典：
                success           模型有没有返回有效的文字
                text              回复文字（已去掉<think>思考内容）
                json / json_ok    提取出来的JSON对象 / 是否提取成功（只有传了json_format才有意义）
                error             失败原因，成功时是None
                truncated         是不是因为到了max_tokens被截断
                prompt_tokens     输入的token数（Ollama统计的真实值）
                completion_tokens 输出的token数
                elapsed_seconds   耗时
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        result = {
            "success": False, "text": "", "json": None, "json_ok": False, "error": None,
            "truncated": False, "prompt_tokens": 0, "completion_tokens": 0,
            "elapsed_seconds": 0.0, "model": self.model_name,
        }

        # 初始化的时候没连上，可能是后来才启动了Ollama，这里再试一次
        if not self.available:
            self.test_connection()
        if not self.available:
            result["error"] = self.connection_message
            return result

        payload = {
            "model": self.model_name,
            "messages": self._build_messages(prompt, system_prompt, json_format),
            "stream": False,
            "think": False,     # 关掉qwen3的思考模式（旧版本Ollama不认这个字段会直接忽略）
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }

        start_time = time.time()
        try:
            response = self._request("POST", "/api/chat", payload)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            result["error"] = f"Ollama返回了错误 HTTP {e.code}: {body}"
        except Exception as e:
            # 超时、连接断开、返回内容解析不了，都会到这里
            result["error"] = f"请求失败：{type(e).__name__}: {e}"
        result["elapsed_seconds"] = time.time() - start_time
        if result["error"]:
            return result

        raw_text = (response.get("message") or {}).get("content", "")
        text = self._clean_text(raw_text)
        result["text"] = text
        result["prompt_tokens"] = response.get("prompt_eval_count", 0) or 0
        result["completion_tokens"] = response.get("eval_count", 0) or 0
        result["truncated"] = response.get("done_reason") == "length"

        if not text:
            result["error"] = "模型返回的内容为空（可能是思考内容把max_tokens用完了）"
            return result

        result["success"] = True
        if json_format:
            result["json"], result["json_ok"] = self.extract_json(text)
        return result

    def generate_batch(self, requests, system_prompt=None, temperature=None, max_tokens=None,
                       json_format=None, verbose=True):
        """
        批量生成：依次处理多个请求，返回结果列表（顺序和输入一致）。

        requests里的每一项可以是字符串（就是prompt），也可以是字典（prompt必须有，
        system_prompt/temperature/max_tokens/json_format可以单独指定，没写的用这个函数的参数）。
        本地Ollama一次只处理一个请求，所以这里是一个一个顺序跑，不是并行。
        """
        results = []
        total = len(requests)
        for i, item in enumerate(requests):
            if isinstance(item, str):
                item = {"prompt": item}

            result = self.generate(
                prompt=item["prompt"],
                system_prompt=item.get("system_prompt", system_prompt),
                temperature=item.get("temperature", temperature),
                max_tokens=item.get("max_tokens", max_tokens),
                json_format=item.get("json_format", json_format),
            )
            result["index"] = i
            results.append(result)

            if verbose:
                status = "成功" if result["success"] else f"失败（{result['error']}）"
                print(f"  批量生成 [{i + 1}/{total}] {status}，耗时{result['elapsed_seconds']:.1f}秒，"
                      f"输出{result['completion_tokens']}个token")
        return results


if __name__ == "__main__":
    llm = LLMGenerator()
    print(llm.connection_message)
    if not llm.available:
        raise SystemExit(1)

    print("\n--- 测试1：普通生成 ---")
    r = llm.generate("用一句话说明二甲双胍是什么药。", system_prompt="你是一名医学助手，回答简洁准确。",
                     temperature=0.1, max_tokens=100)
    print("成功:", r["success"], " 耗时: %.1f秒" % r["elapsed_seconds"],
          " 输入/输出token:", r["prompt_tokens"], "/", r["completion_tokens"])
    print("回复:", r["text"])

    print("\n--- 测试2：要求JSON输出 ---")
    r = llm.generate("二甲双胍是否可用于治疗2型糖尿病？", system_prompt="你是一名医学助手。",
                     temperature=0.1, max_tokens=150,
                     json_format={"answer": "是/否", "reason": "一句话理由"})
    print("原始回复:", r["text"])
    print("提取成功:", r["json_ok"], " 结果:", r["json"])

    print("\n--- 测试3：批量生成 ---")
    batch = llm.generate_batch(["阿司匹林的主要作用是什么？一句话回答。", "他汀类药物的主要作用是什么？一句话回答。"],
                               temperature=0.1, max_tokens=80)
    for r in batch:
        print(f"[{r['index']}]", r["text"])
