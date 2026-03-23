"""
记忆系统提示词模板
从 prompts/memory/*.txt 加载，支持 .format() 填充变量
"""

from prompts import PromptLoader

_loader = PromptLoader()

# 交互记录总结 prompt
# 变量：{input}, {response}
INTERACTION_SUMMARY_PROMPT = _loader.load("memory/interaction_summary.txt")

# 定时汇总 prompt
# 变量：{active_memories}, {recent_interactions}
PERIODIC_SUMMARY_PROMPT = _loader.load("memory/periodic_summary.txt")

# 自我记忆提取 prompt
# 变量：{input}, {response}
STANCE_EXTRACTION_PROMPT = _loader.load("memory/stance_extraction.txt")

# 用户记忆结构化抽取 prompt
# 变量：{comments}, {ai_response}
VIEWER_SUMMARY_PROMPT = _loader.load("memory/viewer_summary.txt")
