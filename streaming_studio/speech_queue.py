"""
语音调度队列
将 LLM 生成和 TTS 播放解耦，通过带优先级和 TTL 的队列实现单句调度。

优先级（越小越优先）：
  0 — SC / 上舰 / >=5元礼物（付费事件）
  1 — 普通弹幕回复 / CommentSession 续接
  2 — 小礼物 / 入场问候
  3 — 视频解说 / 独白
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .models import StreamerResponse, Comment


# ── 优先级常量 ──

PRIORITY_PAID = 0
PRIORITY_DANMAKU = 1
PRIORITY_EVENT_LOW = 2
PRIORITY_VIDEO = 3


@dataclass
class SpeechItem:
  """队列中的单句条目"""
  segment: dict
  priority: int
  ttl: float
  source: str
  response_id: str
  response: StreamerResponse
  segment_index: int = 0
  segment_total: int = 1
  comments: list[Comment] = field(default_factory=list)
  generated_at: float = field(default_factory=time.monotonic)
  id: str = field(default_factory=lambda: str(uuid.uuid4()))

  @property
  def expired(self) -> bool:
    return (time.monotonic() - self.generated_at) > self.ttl

  @property
  def age(self) -> float:
    return time.monotonic() - self.generated_at

  @property
  def is_last_segment(self) -> bool:
    return self.segment_index >= max(self.segment_total - 1, 0)


class SpeechQueue:
  """
  带优先级和 TTL 的语音调度队列。

  排序规则：(priority, generated_at) — 同优先级先到先出。
  队列满时驱逐优先级最低的条目（priority 数值最大、最旧的）。
  pop 时自动跳过过期条目。
  """

  def __init__(self, max_size: int = 4):
    self._items: list[SpeechItem] = []
    self._max_size = max_size
    self._new_item = asyncio.Event()
    self._space_available = asyncio.Event()
    self._space_available.set()
    self._lock = asyncio.Lock()

    self._total_pushed = 0
    self._total_played = 0
    self._total_expired = 0
    self._total_evicted = 0

  @property
  def size(self) -> int:
    return len(self._items)

  @property
  def is_empty(self) -> bool:
    return len(self._items) == 0

  @property
  def available_slots(self) -> int:
    return max(0, self._max_size - len(self._items))

  async def push(self, item: SpeechItem) -> list[SpeechItem]:
    """
    入队。队满时驱逐优先级最低的条目。

    Returns:
      被驱逐的条目列表
    """
    async with self._lock:
      self._items = [i for i in self._items if not i.expired]
      self._total_expired += 0  # expired ones cleaned silently

      evicted: list[SpeechItem] = []
      while len(self._items) >= self._max_size:
        candidate_pool = [
          queued for queued in self._items
          if queued.response_id != item.response_id
        ] or list(self._items)
        worst = max(candidate_pool, key=lambda i: (i.priority, i.generated_at))
        if worst.priority >= item.priority:
          self._items.remove(worst)
          evicted.append(worst)
          self._total_evicted += 1
        else:
          break

      if len(self._items) < self._max_size:
        self._items.append(item)
        self._items.sort(key=lambda i: (i.priority, i.generated_at))
        self._total_pushed += 1

      self._new_item.set()
      if len(self._items) < self._max_size:
        self._space_available.set()
      else:
        self._space_available.clear()

      return evicted

  async def pop(self) -> Optional[SpeechItem]:
    """取出最高优先级的未过期条目。"""
    async with self._lock:
      while self._items:
        item = self._items.pop(0)
        if not item.expired:
          self._total_played += 1
          self._space_available.set()
          return item
        self._total_expired += 1

      self._new_item.clear()
      self._space_available.set()
      return None

  async def flush_source(self, source: str) -> list[SpeechItem]:
    """清空指定 source 的所有待播条目。"""
    async with self._lock:
      flushed = [i for i in self._items if i.source == source]
      self._items = [i for i in self._items if i.source != source]
      if len(self._items) < self._max_size:
        self._space_available.set()
      return flushed

  async def flush_all(self) -> list[SpeechItem]:
    """清空所有待播条目。"""
    async with self._lock:
      flushed = list(self._items)
      self._items.clear()
      self._space_available.set()
      return flushed

  async def peek(self) -> Optional[SpeechItem]:
    """查看队首条目但不取出。"""
    async with self._lock:
      for item in self._items:
        if not item.expired:
          return item
      return None

  async def wait_for_item(self) -> None:
    """阻塞等待直到有新条目入队。"""
    self._new_item.clear()
    await self._new_item.wait()

  async def wait_for_space(self) -> None:
    """阻塞等待直到队列有空位。"""
    self._space_available.clear()
    await self._space_available.wait()

  def debug_state(self) -> dict:
    items_preview = []
    for item in self._items:
      items_preview.append({
        "source": item.source,
        "priority": item.priority,
        "segment": f"{item.segment_index + 1}/{item.segment_total}",
        "age": round(item.age, 1),
        "ttl": item.ttl,
        "expired": item.expired,
        "text_zh": item.segment.get("text_zh", "")[:40],
      })
    return {
      "size": len(self._items),
      "max_size": self._max_size,
      "items": items_preview,
      "stats": {
        "total_pushed": self._total_pushed,
        "total_played": self._total_played,
        "total_expired": self._total_expired,
        "total_evicted": self._total_evicted,
      },
    }
