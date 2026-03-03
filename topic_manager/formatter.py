"""
话题输出格式化
将话题表转换为 prompt 注入文本（无模型请求）
"""

import logging
import random
from datetime import datetime
from enum import Enum
from typing import Optional

from prompts import PromptLoader
from .config import TopicManagerConfig
from .models import Topic
from .table import TopicTable

logger = logging.getLogger(__name__)

_loader = PromptLoader()
_STALE_INSTRUCTION = _loader.load("topic/stale_instruction.txt")
_FOLLOWUP_INSTRUCTION = _loader.load("topic/followup_instruction.txt")
_PROACTIVE_CONTINUATION = _loader.load("topic/proactive_continuation.txt")
_ASK_AUDIENCE_INSTRUCTION = _loader.load("topic/ask_audience_instruction.txt")
_NEW_TOPIC_INSTRUCTION = _loader.load("topic/new_topic_instruction.txt")


class ProactiveSpeakStrategy(str, Enum):
  """主动发言策略"""
  CONTINUE_TOPIC = "continue"  # 继续推进当前话题
  ASK_AUDIENCE   = "ask"       # 向观众提问
  NEW_TOPIC      = "new"       # 挑起新话题


def _significance_label(sig: float) -> str:
  """将 significance 数值转换为自然语言评价"""
  if sig >= 0.8:
    return "很高"
  if sig >= 0.6:
    return "较高"
  if sig >= 0.4:
    return "中等"
  if sig >= 0.2:
    return "较低"
  return "很低"


def get_annotations(table: TopicTable) -> dict[str, str]:
  """
  获取弹幕 → 话题标题的标注映射

  Args:
    table: 话题表

  Returns:
    dict[comment_id, topic_title]（自然语言标题）
  """
  annotations: dict[str, str] = {}
  for topic in table.get_all():
    for cid in topic.comment_ids:
      annotations[cid] = topic.title
  return annotations


def format_topic_context(
  table: TopicTable,
  new_comments_ids: list[str],
  config: TopicManagerConfig,
) -> str:
  """
  格式化话题上下文（注入 prompt）

  包含两部分：
  (a) 话题摘要（top_k + 最近弹幕提及的话题）
  (b) 额外指令（过期话题、跟进建议、冷场建议）

  弹幕-话题对应关系通过 get_annotations() 直接标注在弹幕上，此处不重复。

  Args:
    table: 话题表
    new_comments_ids: 本轮新弹幕的 ID 列表
    config: 配置

  Returns:
    格式化的话题上下文文本
  """
  if table.count() == 0:
    return ""

  # 确定要展示的话题
  topics_to_show = _select_topics(table, new_comments_ids, config)
  if not topics_to_show:
    return ""

  parts = []

  # (a) 话题摘要
  topic_summary = _format_topic_summary(topics_to_show, config)
  if topic_summary:
    parts.append(topic_summary)

  # (b) 额外指令
  instructions = _format_instructions(topics_to_show, new_comments_ids, table, config)
  if instructions:
    parts.append(instructions)

  return "\n\n".join(parts)


def _select_topics(
  table: TopicTable,
  new_comment_ids: list[str],
  config: TopicManagerConfig,
) -> list[Topic]:
  """
  选择要展示的话题

  规则：top_k + 最近弹幕提及的话题（即使 significance 低）

  Args:
    table: 话题表
    new_comment_ids: 本轮新弹幕 ID
    config: 配置

  Returns:
    要展示的话题列表（去重）
  """
  # top_k 话题
  top_topics = table.get_top_k(config.top_k_topics)
  shown_ids = {t.topic_id for t in top_topics}

  # 最近弹幕提及的话题（即使 significance 低）
  for cid in new_comment_ids:
    topic = table.get_by_comment(cid)
    if topic and topic.topic_id not in shown_ids:
      top_topics.append(topic)
      shown_ids.add(topic.topic_id)

  return top_topics


def _format_topic_summary(
  topics: list[Topic],
  config: TopicManagerConfig,
) -> str:
  """格式化话题摘要"""
  now = datetime.now()
  lines = ["【当前话题】"]

  for topic in topics:
    label = _significance_label(topic.significance)
    lines.append(f"\n--- {topic.title} (重要性: {label}) ---")
    lines.append(f"进度: {topic.topic_progress}")

    if topic.suggestion:
      lines.append(f"建议: {topic.suggestion}")

    if topic.stale:
      lines.append("(!) 这个话题已经聊了很久")

    # 空闲时长标注
    idle_seconds = (now - topic.last_discussed_at).total_seconds()
    if idle_seconds >= 60:
      idle_minutes = int(idle_seconds // 60)
      lines.append(f"(i) 已有约 {idle_minutes} 分钟无人讨论此话题")
    elif idle_seconds >= 20:
      lines.append(f"(i) 已有约 {int(idle_seconds)} 秒无人讨论此话题")

  return "\n".join(lines)


def _pick_proactive_strategy(
  topics: list[Topic],
  config: TopicManagerConfig,
) -> tuple[ProactiveSpeakStrategy, Optional[Topic]]:
  """
  根据话题状态加权随机选择主动发言策略

  Returns:
    (strategy, topic) — NEW_TOPIC 时 topic 为 None
  """
  active = [
    t for t in topics
    if not t.stale and t.significance >= config.proactive_topic_min_significance
  ]

  if not active:
    return ProactiveSpeakStrategy.NEW_TOPIC, None

  best = max(active, key=lambda t: t.significance)
  strategy = random.choices(
    [
      ProactiveSpeakStrategy.CONTINUE_TOPIC,
      ProactiveSpeakStrategy.ASK_AUDIENCE,
      ProactiveSpeakStrategy.NEW_TOPIC,
    ],
    weights=[0.40, 0.35, 0.25],
  )[0]

  return strategy, (None if strategy == ProactiveSpeakStrategy.NEW_TOPIC else best)


def _format_instructions(
  topics: list[Topic],
  new_comment_ids: list[str],
  table: TopicTable,
  config: TopicManagerConfig,
) -> str:
  """格式化额外指令"""
  instructions = []

  # 过期话题指令
  stale_topics = [t for t in topics if t.stale]
  if stale_topics:
    names = "、".join(f"「{t.title}」" for t in stale_topics)
    instructions.append(_STALE_INSTRUCTION.format(names=names))

  # 冷场 / 弹幕稀疏时的话题跟进建议
  comment_count = len(new_comment_ids)
  if comment_count <= config.sparse_comment_threshold:
    if comment_count == 0:
      # 完全没弹幕：三选一策略（继续话题 / 向观众提问 / 挑起新话题）
      strategy, pick = _pick_proactive_strategy(topics, config)
      if strategy == ProactiveSpeakStrategy.CONTINUE_TOPIC and pick and pick.suggestion:
        instructions.append(
          _FOLLOWUP_INSTRUCTION.format(title=pick.title, suggestion=pick.suggestion)
        )
      elif strategy == ProactiveSpeakStrategy.ASK_AUDIENCE and pick:
        instructions.append(
          _ASK_AUDIENCE_INSTRUCTION.format(title=pick.title)
        )
      else:
        instructions.append(_NEW_TOPIC_INSTRUCTION)
    else:
      # 有少量弹幕但很稀疏：温和推进（保持原有逻辑）
      followup_topics = [t for t in topics if t.suggestion and not t.stale]
      if followup_topics:
        best = max(followup_topics, key=lambda t: t.significance)
        instructions.append(
          _PROACTIVE_CONTINUATION.format(title=best.title, suggestion=best.suggestion)
        )

  if not instructions:
    return ""

  return "【话题指令】\n" + "\n".join(f"- {i}" for i in instructions)
