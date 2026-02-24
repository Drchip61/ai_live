"""
记忆格式化
将跨层检索结果格式化为可嵌入 prompt 的文本，按层级和时间排列
"""

from datetime import datetime
from typing import Optional

from .layers.base import MemoryEntry


def _relative_time(memory_time: datetime, now: datetime) -> str:
  """
  计算相对时间描述

  Args:
    memory_time: 记忆时间戳
    now: 当前时间

  Returns:
    相对时间字符串，如 "1分34秒之前" 或 "25分前"
  """
  delta = now - memory_time
  total_seconds = max(0, int(delta.total_seconds()))

  if total_seconds < 60:
    return f"{total_seconds}秒之前"

  minutes = total_seconds // 60
  seconds = total_seconds % 60

  if seconds > 0 and minutes < 10:
    # 短时间：精确到秒
    return f"{minutes}分{seconds}秒之前"
  else:
    # 长时间：只到分钟
    return f"{minutes}分前"


def _truncate_response(response: str, max_length: int) -> str:
  """截断回复原文，超过 max_length 时末尾加省略号"""
  if len(response) <= max_length:
    return response
  return response[:max_length] + "..."


def format_active_memories(
  entries: list[MemoryEntry],
  include_response: bool = True,
  response_max_length: int = 80,
) -> str:
  """
  格式化 active 层记忆（无 RAG，直接时序列出）

  Args:
    entries: active 层记忆列表
    include_response: 是否引用主播回复原文
    response_max_length: 回复原文最大显示长度

  Returns:
    格式化文本
  """
  if not entries:
    return ""

  lines = ["【近期记忆】"]
  for entry in entries:
    lines.append(f"- {entry.content}")
    if include_response and entry.metadata:
      response = entry.metadata.get("response", "")
      if response:
        truncated = _truncate_response(response, response_max_length)
        lines.append(f"  → 我说：「{truncated}」")
  return "\n".join(lines)


def format_retrieved_memories(
  entries: list[MemoryEntry],
  now: Optional[datetime] = None,
  current_session_id: Optional[str] = None,
  include_temp_response: bool = False,
  response_max_length: int = 80,
) -> str:
  """
  格式化跨层 RAG 检索结果（按层级分组，各层有小标题）

  排列规则：
  1. 层级从短到长：temporary → summary → static
  2. 同层级内时间从近到远

  各层小标题和格式：
  - temporary: 【相关短期回忆】 + 相对时间前缀
  - summary: 【相关长期回忆】 + 相对时间前缀
  - static: 【关于自己】 + category 前缀（已在 StaticLayer 中处理）

  跨会话标注：
  - 如果 current_session_id 有值，且记忆的 session_id 与之不同，
    添加【来自之前的直播】标注

  Args:
    entries: 跨层检索结果
    now: 当前时间（默认 datetime.now()）
    current_session_id: 当前直播会话 ID（用于跨会话标注）
    include_temp_response: 是否在 temporary 层引用主播回复原文
    response_max_length: 回复原文最大显示长度

  Returns:
    格式化文本，无结果时返回空字符串
  """
  if not entries:
    return ""

  if now is None:
    now = datetime.now()

  # 按层级分组
  layer_groups: dict[str, list[MemoryEntry]] = {}
  for entry in entries:
    layer_groups.setdefault(entry.layer, []).append(entry)

  # 各层内按时间从近到远排序
  for layer_entries in layer_groups.values():
    layer_entries.sort(key=lambda e: e.timestamp, reverse=True)

  # 层级顺序和标题
  layer_spec = [
    ("temporary", "【相关短期回忆】"),
    ("summary", "【相关长期回忆】"),
    ("static", "【关于自己】"),
  ]

  parts = []
  for layer_name, header in layer_spec:
    layer_entries = layer_groups.get(layer_name)
    if not layer_entries:
      continue

    lines = [header]
    for entry in layer_entries:
      # 检查是否来自之前的直播
      cross_session_prefix = ""
      if current_session_id is not None and entry.metadata:
        mem_session = entry.metadata.get("session_id")
        if mem_session is not None and mem_session != current_session_id:
          cross_session_prefix = "【来自之前的直播】"

      if entry.layer == "static":
        # static 层已带 category 前缀
        lines.append(f"- {cross_session_prefix}{entry.content}")
      else:
        # temporary / summary 层加相对时间前缀
        rel_time = _relative_time(entry.timestamp, now)
        lines.append(f"- {cross_session_prefix}【{rel_time}的记忆】{entry.content}")

        # temporary 层可选引用回复原文
        if include_temp_response and entry.layer == "temporary" and entry.metadata:
          response = entry.metadata.get("response", "")
          if response:
            truncated = _truncate_response(response, response_max_length)
            lines.append(f"  → 我说：「{truncated}」")

    parts.append("\n".join(lines))

  return "\n\n".join(parts)
