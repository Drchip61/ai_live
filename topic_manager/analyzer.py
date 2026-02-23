"""
回复后分析器
多个并行异步任务，返回 delta 建议
"""

import asyncio
import json
import logging
from typing import Optional

from langchain_core.language_models import BaseChatModel

from .config import TopicManagerConfig
from .models import Topic, ContentAnalysisDelta, RhythmAnalysisDelta
from .prompts import CONTENT_ANALYSIS_PROMPT, RHYTHM_ANALYSIS_PROMPT
from .table import TopicTable

logger = logging.getLogger(__name__)


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


def _format_topic_table(table: TopicTable) -> str:
  """格式化话题表供 prompt 使用"""
  topics = table.get_all()
  if not topics:
    return "（当前无话题）"
  lines = []
  for t in topics:
    stale_mark = " [过期]" if t.stale else ""
    label = _significance_label(t.significance)
    lines.append(
      f"- {t.topic_id}「{t.title}」(重要性: {label}{stale_mark}): "
      f"{t.topic_progress}"
    )
    if t.suggestion:
      lines.append(f"  建议: {t.suggestion}")
  return "\n".join(lines)


def _parse_json_response(text: str) -> Optional[dict]:
  """
  解析 LLM 返回的 JSON（容错处理）

  Args:
    text: LLM 原始输出

  Returns:
    解析后的 dict，失败返回 None
  """
  text = text.strip()
  # 清理 markdown 代码块
  if text.startswith("```"):
    text = text.split("\n", 1)[-1]
  if text.endswith("```"):
    text = text.rsplit("```", 1)[0]
  text = text.strip()

  try:
    return json.loads(text)
  except json.JSONDecodeError:
    logger.error("JSON 解析失败: %s", text[:200])
    return None


async def analyze_content(
  table: TopicTable,
  recent_comments: str,
  response: str,
  model: BaseChatModel,
  config: TopicManagerConfig,
) -> ContentAnalysisDelta:
  """
  内容分析（任务 A）

  判断话题进度、新话题、跟进建议。

  Args:
    table: 话题表
    recent_comments: 格式化的最近弹幕文本
    response: 主播回复文本
    model: 小模型
    config: 配置

  Returns:
    ContentAnalysisDelta
  """
  try:
    prompt = CONTENT_ANALYSIS_PROMPT.format(
      topic_table=_format_topic_table(table),
      recent_comments=recent_comments,
      response=response,
    )
    result = await model.ainvoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)

    data = _parse_json_response(text)
    if data is None:
      return ContentAnalysisDelta()

    # 解析新话题
    new_topics = []
    for item in data.get("new_topics", []):
      tid = item.get("topic_id", "").strip()
      title = item.get("title", "").strip() or tid
      if tid and table.get(tid) is None:
        new_topics.append(Topic(
          topic_id=tid,
          title=title,
          significance=config.initial_significance,
          topic_progress=item.get("progress", "新话题"),
          suggestion=item.get("suggestion", ""),
        ))

    return ContentAnalysisDelta(
      progress_updates=data.get("progress_updates", {}),
      new_topics=new_topics,
      suggestion_updates=data.get("suggestion_updates", {}),
    )

  except Exception as e:
    logger.error("内容分析失败: %s", e)
    return ContentAnalysisDelta()


async def analyze_rhythm(
  table: TopicTable,
  recent_comments: str,
  response: str,
  model: BaseChatModel,
) -> RhythmAnalysisDelta:
  """
  节奏分析（任务 B）

  判断过期话题、建议等待时间。
  需要深度理解回复正文内容，不仅看频率。

  Args:
    table: 话题表
    recent_comments: 格式化的最近弹幕文本
    response: 主播回复文本
    model: 小模型

  Returns:
    RhythmAnalysisDelta
  """
  try:
    prompt = RHYTHM_ANALYSIS_PROMPT.format(
      topic_table=_format_topic_table(table),
      recent_comments=recent_comments,
      response=response,
    )
    result = await model.ainvoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)

    data = _parse_json_response(text)
    if data is None:
      return RhythmAnalysisDelta()

    stale_ids = data.get("stale_topic_ids", [])
    min_wait = data.get("suggested_min_wait")
    max_wait = data.get("suggested_max_wait")

    timing = None
    if min_wait is not None and max_wait is not None:
      timing = (float(min_wait), float(max_wait))

    return RhythmAnalysisDelta(
      stale_topic_ids=stale_ids,
      suggested_timing=timing,
    )

  except Exception as e:
    logger.error("节奏分析失败: %s", e)
    return RhythmAnalysisDelta()


async def post_reply_analysis(
  table: TopicTable,
  recent_comments: str,
  response: str,
  model: BaseChatModel,
  config: TopicManagerConfig,
) -> tuple[ContentAnalysisDelta, RhythmAnalysisDelta]:
  """
  回复后完整分析（2 个并行异步任务）

  Args:
    table: 话题表
    recent_comments: 格式化的最近弹幕文本
    response: 主播回复文本
    model: 小模型
    config: 配置

  Returns:
    (content_delta, rhythm_delta)
  """
  content_delta, rhythm_delta = await asyncio.gather(
    analyze_content(table, recent_comments, response, model, config),
    analyze_rhythm(table, recent_comments, response, model),
  )
  return content_delta, rhythm_delta
