"""
弹幕分类器
将弹幕分配到已有话题（不创建新话题）
支持单条和批量两种模式，含规则匹配降级策略
"""

import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from .models import Topic
from .prompts import SINGLE_CLASSIFY_PROMPT, BATCH_CLASSIFY_PROMPT
from .table import TopicTable

logger = logging.getLogger(__name__)


def rule_match(
  content: str,
  topics: list[Topic],
) -> Optional[str]:
  """
  规则匹配（第一层，零成本）

  检查弹幕内容与话题 topic_id / progress 的关键词重叠。

  Args:
    content: 弹幕内容
    topics: 当前话题列表

  Returns:
    匹配的 topic_id，未匹配返回 None
  """
  content_lower = content.lower().strip()
  if not content_lower:
    return None

  best_topic_id = None
  best_score = 0

  for topic in topics:
    score = 0
    # topic_id 中的词（snake_case 分割）
    id_words = topic.topic_id.split("_")
    for word in id_words:
      if word and word in content_lower:
        score += 2

    # title 中的关键词
    for char in topic.title:
      if char and char in content_lower:
        score += 1

    # progress 中的关键词
    progress_chars = set(topic.topic_progress)
    overlap = sum(1 for c in content_lower if c in progress_chars)
    # 至少 3 个字符重叠才算
    if overlap >= 3:
      score += overlap / len(content_lower) if content_lower else 0

    if score > best_score:
      best_score = score
      best_topic_id = topic.topic_id

  # 阈值：至少有一定匹配度
  if best_score >= 2:
    return best_topic_id
  return None


def _format_topic_list(topics: list[Topic]) -> str:
  """格式化话题列表供 prompt 使用"""
  if not topics:
    return "（当前无话题）"
  lines = []
  for t in topics:
    lines.append(f"- {t.topic_id}「{t.title}」: {t.topic_progress}")
  return "\n".join(lines)


async def single_classify(
  content: str,
  comment_id: str,
  table: TopicTable,
  model: BaseChatModel,
) -> Optional[str]:
  """
  单条弹幕分类

  先尝试规则匹配，失败时调用小模型。

  Args:
    content: 弹幕内容
    comment_id: 弹幕 ID
    table: 话题表
    model: 小模型

  Returns:
    匹配的 topic_id，未匹配返回 None
  """
  topics = table.get_all()
  if not topics:
    return None

  # 第一层：规则匹配
  matched = rule_match(content, topics)
  if matched:
    logger.debug("规则匹配弹幕 %s → %s", comment_id[:8], matched)
    return matched

  # 第二层：小模型分类
  try:
    prompt = SINGLE_CLASSIFY_PROMPT.format(
      topic_list=_format_topic_list(topics),
      comment_content=content,
    )
    result = await model.ainvoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip().lower()

    if text == "none" or not text:
      return None

    # 验证返回的 topic_id 是否存在
    if table.get(text):
      logger.debug("模型分类弹幕 %s → %s", comment_id[:8], text)
      return text

    logger.debug("模型返回了无效话题 ID: %s", text)
    return None
  except Exception as e:
    logger.error("单条分类失败: %s", e)
    return None


async def batch_classify(
  comments: list[tuple[str, str, str]],
  table: TopicTable,
  model: BaseChatModel,
) -> dict[str, str]:
  """
  批量弹幕分类

  先用规则匹配处理能处理的，剩余的发给模型。

  Args:
    comments: [(comment_id, user_id, content), ...] 列表
    table: 话题表
    model: 小模型

  Returns:
    {comment_id: topic_id} 映射（只包含成功分类的）
  """
  topics = table.get_all()
  if not topics:
    return {}

  results: dict[str, str] = {}
  unmatched: list[tuple[int, str, str]] = []  # (index, comment_id, content)

  # 第一层：规则匹配
  for idx, (cid, _uid, content) in enumerate(comments):
    matched = rule_match(content, topics)
    if matched:
      results[cid] = matched
    else:
      unmatched.append((idx + 1, cid, content))

  if not unmatched:
    return results

  # 第二层：剩余发给模型
  try:
    comments_text = "\n".join(
      f"{idx}. {content}" for idx, _cid, content in unmatched
    )
    prompt = BATCH_CLASSIFY_PROMPT.format(
      topic_list=_format_topic_list(topics),
      comments=comments_text,
    )
    result = await model.ainvoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip()

    # 尝试解析 JSON
    # 清理可能的 markdown 代码块标记
    if text.startswith("```"):
      text = text.split("\n", 1)[-1]
    if text.endswith("```"):
      text = text.rsplit("```", 1)[0]
    text = text.strip()

    mapping = json.loads(text)

    # 将编号映射回 comment_id
    idx_to_cid = {str(idx): cid for idx, cid, _ in unmatched}
    for idx_str, topic_id in mapping.items():
      cid = idx_to_cid.get(idx_str)
      if cid and topic_id != "none" and table.get(topic_id):
        results[cid] = topic_id

  except (json.JSONDecodeError, Exception) as e:
    logger.error("批量分类失败: %s", e)

  return results
