"""
生成模块与提示词工程 第一部分 - 医学提示词模板

生成答案这一步没有设计成"一次生成就完事"，而是拆成四个阶段接力，每个阶段是一次
独立的LLM调用，用各自的system_prompt/user_prompt_template/temperature组合：

1. evidence_evaluator（证据评估器）：先看看14_context_assembler.py拼出来的证据
   够不够、对不对题，不够的话要老实说不够，不要硬凑。
2. answer_generator（答案生成器）：基于评估过的证据生成一版草稿答案，要求每句话
   都能在证据里找到依据，并标注来源。
3. critical_reviewer（批判性审查器）：专门挑草稿答案里站不住脚的地方——有没有编造的
   内容、有没有把机制研究的结论说成临床结论、来源标注对不对、有没有漏掉重要的限制条件。
4. final_assembler（最终组装器）：把审查意见和草稿答案合并成最终版本。

这几个阶段串起来跑一遍，比一次性让模型直接回答要慢，但多了一层自我核查，
应该能减少答案里出现证据之外的编造内容（但这个假设目前还没有真的接上生成模型
跑过实测，只是提示词先写好）。

在整个流程里是第15步。这个文件里只有数据（PromptStage数据类和PROMPT_STAGES字典），
没有可执行的逻辑，所以不能单独运行，由16_verify_generation_prep.py、
20_medical_generation_pipeline.py等脚本用importlib导入后取模板用。
每个阶段的user_prompt_template里有{question}、{context}、{evidence_evaluation}、
{draft_answer}、{review_feedback}这些占位符（各阶段用到的不一样），由调用的一方负责填进去。
依赖：只用了标准库的dataclasses
"""

from dataclasses import dataclass


@dataclass
class PromptStage:
    """一个阶段的提示词配置：显示名称、system提示词、user提示词模板、采样温度、最大输出token数"""
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int


# 四个阶段的配置，key是阶段名。评估和审查这两个只做判断，temperature设得低（0.1），
# 答案生成稍微放开一点（0.3），最终整合是0.2
PROMPT_STAGES = {
    "evidence_evaluator": PromptStage(
        name="证据评估器",
        system_prompt=(
            "你是一名医学文献评估助手。你的任务不是回答问题，而是判断下面提供的检索"
            "证据片段，对回答这个医学问题是否足够、是否切题。请逐条评估每个证据片段：\n"
            "1. 相关性：片段内容是不是真的在讨论问题里提到的那个具体药物/疾病/机制，"
            "还是只是主题上沾点边\n"
            "2. 证据强度：是系统综述/临床试验这类比较可靠的证据，还是机制研究/动物"
            "实验/个案报告这类强度较弱的证据\n"
            "3. 一致性：证据片段之间有没有明显冲突或矛盾的地方\n"
            "如果现有证据不足以回答问题，必须明确指出缺口具体在哪里，不要替用户编造"
            "或过度推断没有依据的结论。"
        ),
        user_prompt_template=(
            "问题：{question}\n\n"
            "检索到的证据片段：\n{context}\n\n"
            "请按上面三个维度逐条评估这些证据片段，最后给出一个总体判断：现有证据是否"
            "足以回答这个问题（充分 / 部分充分 / 不充分），并简要说明理由。"
        ),
        temperature=0.1,
        max_tokens=800,
    ),
    "answer_generator": PromptStage(
        name="答案生成器",
        system_prompt=(
            "你是一名医学文献问答助手，基于提供的证据片段回答医学问题。要求：\n"
            "1. 回答必须完全基于给出的证据片段，不要用片段之外的知识去补充具体结论\n"
            "2. 每个关键陈述后面标注来源，格式为[来源: 文献标题]\n"
            "3. 如果证据片段之间有冲突，要在回答里明确指出，不要选择性忽略某一边\n"
            "4. 如果证据不足以完整回答问题，要明确说明还缺什么，不要为了回答完整而硬凑\n"
            "5. 回答最后加一句：本回答基于检索到的文献证据整理生成，不构成医疗建议，"
            "具体诊疗请遵医嘱。"
        ),
        user_prompt_template=(
            "问题：{question}\n\n"
            "可用证据片段：\n{context}\n\n"
            "证据评估结果：\n{evidence_evaluation}\n\n"
            "请基于以上证据生成回答。"
        ),
        temperature=0.3,
        max_tokens=1200,
    ),
    "critical_reviewer": PromptStage(
        name="批判性审查器",
        system_prompt=(
            "你是一名严格的医学内容审查员，负责检查下面这份草稿答案有没有问题。"
            "重点检查这四点：\n"
            "1. 有没有陈述在提供的证据片段里找不到依据（可能是编造或过度推断）\n"
            "2. 有没有把机制研究/动物实验的结论，当成临床结论来表述（证据强度被夸大）\n"
            "3. 引用的来源标注是否跟内容对得上，有没有张冠李戴\n"
            "4. 有没有遗漏证据里提到的重要限制条件（比如样本量小、尚未经大规模临床"
            "试验验证）\n"
            "只需要指出问题、说明具体位置和原因，不需要重写答案。如果没有发现问题，"
            "明确回复：经检查没有发现明显问题。"
        ),
        user_prompt_template=(
            "原始问题：{question}\n\n"
            "可用证据片段：\n{context}\n\n"
            "草稿答案：\n{draft_answer}\n\n"
            "请按上面四个维度审查这份草稿答案。"
        ),
        temperature=0.1,
        max_tokens=600,
    ),
    "final_assembler": PromptStage(
        name="最终组装器",
        system_prompt=(
            "你是最终答案整合助手，会收到一份草稿答案和一份审查意见，任务是生成最终"
            "版本：\n"
            "1. 如果审查意见指出了问题，要在最终答案里修正——去掉没有依据的陈述、"
            "调低被夸大的证据强度表述、补上遗漏的限制条件\n"
            "2. 如果审查意见说没有发现问题，基本保留草稿答案，只做必要的表达润色\n"
            "3. 保留原来的来源标注格式（[来源: 文献标题]）和末尾的免责声明\n"
            "4. 最终答案结构要清晰，可以分点陈述，但不要丢掉证据之间的因果/对比关系"
        ),
        user_prompt_template=(
            "原始问题：{question}\n\n"
            "草稿答案：\n{draft_answer}\n\n"
            "审查意见：\n{review_feedback}\n\n"
            "请生成修正后的最终答案。 "
        ),
        temperature=0.2,
        max_tokens=1200,
    ),
}