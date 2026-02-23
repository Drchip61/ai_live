"""
话题管理器提示词模板
从 prompts/topic/*.txt 加载，支持 .format() 填充变量
"""

from prompts import PromptLoader

_loader = PromptLoader()

# 单条弹幕分类 prompt
# 变量：{topic_list}, {comment_content}
SINGLE_CLASSIFY_PROMPT = _loader.load("topic/single_classify.txt")

# 批量弹幕分类 prompt
# 变量：{topic_list}, {comments}
BATCH_CLASSIFY_PROMPT = _loader.load("topic/batch_classify.txt")

# 回复后内容分析 prompt（任务 A）
# 变量：{topic_table}, {recent_comments}, {response}
CONTENT_ANALYSIS_PROMPT = _loader.load("topic/content_analysis.txt")

# 回复后节奏分析 prompt（任务 B）
# 变量：{topic_table}, {recent_comments}, {response}
RHYTHM_ANALYSIS_PROMPT = _loader.load("topic/rhythm_analysis.txt")
