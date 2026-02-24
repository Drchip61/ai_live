"""
active 层 — 最短期记忆
FIFO 队列，每次回复后由小模型总结生成，超出容量后溢出到 temporary 层
"""

import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..config import ActiveConfig


@dataclass(frozen=True)
class ActiveMemory:
  """active 层记忆条目"""
  id: str
  content: str
  timestamp: datetime
  response: str = ""  # 主播当时的回复原文


class ActiveLayer:
  """
  active 记忆层

  固定容量的 FIFO 队列。不使用 RAG，按时序直接注入生成上下文。
  溢出的记忆通过回调传递给 temporary 层。
  """

  def __init__(
    self,
    config: Optional[ActiveConfig] = None,
    on_overflow: Optional[callable] = None,
  ):
    """
    初始化 active 层

    Args:
      config: 层配置
      on_overflow: 溢出回调，签名 (content: str, timestamp: datetime, response: str) -> None
                   当旧记忆被挤出时调用
    """
    self._config = config or ActiveConfig()
    self._on_overflow = on_overflow
    self._memories: deque[ActiveMemory] = deque(maxlen=self._config.capacity)

  def add(self, content: str, response: str = "") -> str:
    """
    添加一条记忆

    如果容量已满，最旧的记忆会被挤出并触发 on_overflow 回调。

    Args:
      content: 记忆内容（由小模型总结后的第一人称文本）
      response: 主播当时的回复原文

    Returns:
      记忆 ID
    """
    # 检查是否会溢出
    if len(self._memories) == self._config.capacity and self._on_overflow:
      oldest = self._memories[0]  # 即将被挤出的
      self._on_overflow(oldest.content, oldest.timestamp, oldest.response)

    memory = ActiveMemory(
      id=str(uuid.uuid4()),
      content=content,
      timestamp=datetime.now(),
      response=response,
    )
    self._memories.append(memory)
    return memory.id

  def get_all(self) -> list[ActiveMemory]:
    """
    获取所有记忆（按时序排列，从旧到新）

    Returns:
      记忆列表
    """
    return list(self._memories)

  def count(self) -> int:
    """获取当前记忆数量"""
    return len(self._memories)

  def clear(self) -> None:
    """清空所有记忆"""
    self._memories.clear()
