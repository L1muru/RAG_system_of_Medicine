# 医学文献 RAG 问答系统（课程项目）

用户提一个医学问题，系统先从 PubMed Central 开放获取论文里检索相关文献，再让本地大模型（Ollama 的 qwen3:8b）
基于这些文献作答，并标出每个结论的来源。这个压缩包里是全部代码、说明文档和测试日志。

**规模**：12,135 篇论文，355,695 个文本块；嵌入模型 BAAI/bge-base-en-v1.5，向量库 ChromaDB，
重排序模型 BAAI/bge-reranker-base。运行环境是 Mac M4、16GB 内存、conda 环境 `medrag`。

## 压缩包里有什么

| 内容 | 说明 |
|---|---|
| `00_…py` 到 `27_…py` | 28 个脚本，文件名前的序号就是开发和运行的先后顺序 |
| 9 个 `.md` 说明文档 | 各阶段的设计、结果和问题，总览看 `项目工作总结.md` |
| `generation_logs/` | 真实运行的测试日志和评估指标（`.log`、`.jsonl`） |
| `chunks_output/` | 只放了两个统计文件（`processing_stats.json`、`vector_index_stats.json`） |
| `requirements.txt` | 用到的第三方库（没有固定版本号） |

**没有包含的内容**：原始论文数据（`medrag_data/`）、分块结果（`chunks_output/chunks.parquet`等，约 800MB）、
向量库（`chroma_db/`）、各种中间 pkl 文件（`pkl_model/`）。这些体积太大，也都可以用脚本重新生成。
没有这些文件的话，00 到 09 号脚本需要重新跑一遍，10 号以后的检索和生成脚本才能运行。

## 运行顺序

| 阶段 | 脚本 | 做什么 |
|---|---|---|
| 建库（只做一次） | `00_download_oa_comm.py` | 下载 PMC 开放获取论文 XML |
| | `01_parse_articles.py` | 解析 XML，存成 `parsed_articles.pkl` |
| | `02`～`07` | 数据质量、token 分布、术语、章节结构分析（为分块策略提供依据） |
| | `08_chunk_documents.py` | 分块，输出 `chunks_output/chunks.parquet` |
| | `09_build_vector_index.py` | 向量化并写入 ChromaDB（约 4 小时） |
| 检索 | `10_query_understanding.py` | 查询理解：清洗、识别实体、扩展同义词 |
| | `11_multi_path_retriever.py` | 向量检索 + BM25，融合候选 |
| | `12_reranker.py` | cross-encoder 重排序 |
| | `13_retrieval_pipeline.py` | 把上面几步串起来，可交互测试 |
| 生成 | `14_context_assembler.py`、`15_prompt_templates.py` | 上下文组装、四阶段提示词 |
| | `16`～`18` | 组装器、提示词、真实 tokenizer 的验证和演示 |
| | `19_llm_generator.py` | 调用本地 Ollama |
| | `20_medical_generation_pipeline.py` | 完整生成流水线（证据评估 → 草稿 → 审查 → 最终答案 → 后处理） |
| | `21_test_generation_pipeline.py`、`22_verify_generation_logic.py` | 真实流程测试、离线逻辑验证 |
| 评估、缓存、批量 | `23_answer_evaluator.py` | 答案评估（ROUGE、关键信息召回、幻觉风险、可读性） |
| | `24_generation_cache.py` | 生成缓存 |
| | `25_batch_generator.py` | 线程池批量处理 |
| | `26_verify_eval_cache_batch.py` | 上面三个的离线验证（不需要 Ollama，先运行） |
| | `27_test_evaluation_cache_batch.py` | 用 4 个测试问题的真实测试（需要 Ollama，后运行） |

脚本之间用 `importlib.import_module("14_context_assembler")` 这样的写法互相导入，因为文件名以数字开头，
不能用普通的 `import` 语句。运行时请在项目文件夹里执行。

## 常用命令

```
conda activate medrag
pip install -r requirements.txt

# 不需要 Ollama 的验证
python 22_verify_generation_logic.py              # 生成流程逻辑，64 项
python 26_verify_eval_cache_batch.py              # 评估、缓存、批量，202 项
python 16_verify_generation_prep.py --offline     # 组装器与提示词

# 需要先在另一个终端运行 ollama serve，并下载 qwen3:8b
python 21_test_generation_pipeline.py             # 生成流程真实测试
python 27_test_evaluation_cache_batch.py          # 评估、缓存、批量的真实测试
python 27_test_evaluation_cache_batch.py --quick  # 只测前两个问题
python 27_test_evaluation_cache_batch.py --workers 1   # 指定批量线程数
```

长时间运行时在命令前加 `caffeinate -i`，防止 Mac 休眠（有一次测试出现过约 65 分钟的异常耗时，推测是休眠）。

## 文档索引

| 文档 | 内容 |
|---|---|
| `项目工作总结.md` | 全部工作的总览：流程、技术栈、结果、问题、局限、后续方向 |
| `RAG数据分析与设计说明.md` | 容量评估、数据分析、分块策略的设计依据 |
| `文档解析与分割说明.md` | 解析与分块的流程和当前的统计数字 |
| `向量化与索引构建说明.md` | 向量化与 ChromaDB 索引 |
| `查询理解与增强说明.md` | 查询理解 |
| `多路检索与重排序说明.md` | 混合检索与重排序 |
| `上下文组装与提示词模板说明.md` | 上下文组装与提示词 |
| `本地LLM集成与生成流程说明.md` | 本地大模型与生成流水线 |
| `答案评估_缓存与批量处理说明.md` | 评估、缓存、批量；含任务完成情况、两次真实运行的结果和问题分析 |

## 目前的状态和需要知道的局限

- 所有脚本已完成并验证过：离线验证 202 项和 64 项全部通过，真实测试自动检查 15 项通过（共运行了两次）。
- 检索质量不稳定：BM25 只建了 20,000 个文本块（全库的 5.6%），语料里没有临床指南，测试问题里有一半只能回答"证据不足"。
- 评估器的分数只有参考意义：标准答案是示例，关键信息召回不看否定，幻觉检测只抓绝对化说法。
- 批量处理的加速没有定论：两次真实运行结果不一致（第一次更慢，第二次约快 1.5 倍）。
- 完整的局限和后续改进方向见 `项目工作总结.md` 的第六、七部分。
