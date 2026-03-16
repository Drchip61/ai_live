"""
场景记忆缓存
异步用小模型 VLM 描述视频帧，维护内存滚动缓冲，
为主模型提供时序画面上下文（"之前发生了什么"）
"""

import asyncio
import base64
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from prompts import PromptLoader
from .config import SceneMemoryConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneSnapshot:
  """单条场景描述快照"""
  timestamp_sec: float
  """视频时间位置（秒）"""
  description: str
  """小模型生成的场景描述"""
  created_at: datetime = field(default_factory=datetime.now)
  """系统时间"""


def _decode_to_gray(b64_jpeg: str) -> Optional[np.ndarray]:
  """将 base64 JPEG 解码为灰度图"""
  try:
    raw = base64.b64decode(b64_jpeg)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img
  except Exception:
    return None


class SceneMemoryCache:
  """
  异步场景记忆缓存

  接收 VideoPlayer 帧回调 → 检测画面变化 → 异步调小模型生成文字描述 →
  存入内存滚动缓冲 → 供主模型 prompt 注入时序上下文。

  所有 VLM 调用在后台 asyncio task 中执行，不阻塞主循环。
  """

  def __init__(
    self,
    model: BaseChatModel,
    config: Optional[SceneMemoryConfig] = None,
  ):
    self._model = model
    self._config = config or SceneMemoryConfig()
    self._recent: deque[SceneSnapshot] = deque(maxlen=self._config.buffer_size)

    # 画面变化检测状态
    self._prev_hist: Optional[np.ndarray] = None
    self._last_describe_sec: float = -999.0

    # 加载场景描述 prompt 模板
    loader = PromptLoader()
    self._describe_prompt = loader.load("studio/scene_description.txt")

    # 后台任务引用
    self._background_tasks: set[asyncio.Task] = set()
    self._loop: Optional[asyncio.AbstractEventLoop] = None

    # 并发控制：同时最多一个描述任务在执行
    self._describing = False

  def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
    """设置事件循环引用（在 start() 中调用，避免 on_frame 跨线程问题）"""
    self._loop = loop

  def on_frame(self, frame) -> None:
    """
    帧回调（由 VideoPlayer tick 循环调用，可能在不同线程）

    检测画面变化 + 冷却判断，通过时提交异步描述任务。
    """
    if not self._config.enabled:
      return

    # 已有描述任务在执行，跳过
    if self._describing:
      return

    timestamp_sec = frame.timestamp_sec

    # 冷却检查
    if timestamp_sec - self._last_describe_sec < self._config.min_describe_interval:
      return

    # 画面变化检测
    if not self._frame_changed(frame.base64_jpeg):
      return

    self._last_describe_sec = timestamp_sec
    self._describing = True

    # 在事件循环中创建异步任务
    if self._loop is not None and self._loop.is_running():
      future = asyncio.run_coroutine_threadsafe(
        self._describe_frame(frame.base64_jpeg, timestamp_sec),
        self._loop,
      )
      future.add_done_callback(lambda _: None)

  def _frame_changed(self, b64_jpeg: str) -> bool:
    """
    通过灰度直方图相关性检测画面是否发生显著变化。

    Returns:
      True 表示画面变化显著，应触发描述。
    """
    gray = _decode_to_gray(b64_jpeg)
    if gray is None:
      return False

    new_hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(new_hist, new_hist)

    if self._prev_hist is None:
      self._prev_hist = new_hist
      return True

    corr = cv2.compareHist(self._prev_hist, new_hist, cv2.HISTCMP_CORREL)
    self._prev_hist = new_hist
    return corr < self._config.change_threshold

  async def _describe_frame(self, frame_b64: str, timestamp_sec: float) -> None:
    """用小模型 VLM 生成场景描述，追加到缓冲"""
    try:
      prev_desc = self._recent[-1].description if self._recent else ""
      if prev_desc:
        prev_section = f"上一条描述：{prev_desc}\n"
      else:
        prev_section = ""

      prompt_text = self._describe_prompt.format(prev_section=prev_section)

      content = [
        {
          "type": "image_url",
          "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        },
        {"type": "text", "text": prompt_text},
      ]
      result = await self._model.ainvoke([HumanMessage(content=content)])
      text = result.content if hasattr(result, "content") else str(result)
      text = text.strip()

      if text:
        self._recent.append(SceneSnapshot(
          timestamp_sec=timestamp_sec,
          description=text,
        ))
        logger.debug("场景描述 [%.1fs]: %s", timestamp_sec, text[:60])

    except Exception as e:
      logger.error("场景描述生成失败: %s", e)
    finally:
      self._describing = False

  def to_prompt(self, max_items: int = 0) -> str:
    """
    格式化最近场景描述为 prompt 段落。

    Args:
      max_items: 最大条目数，0 表示使用配置默认值。

    Returns:
      格式化的文字段落，缓冲为空时返回空字符串。
    """
    if not self._recent:
      return ""

    limit = max_items or self._config.max_prompt_items
    items = list(self._recent)[-limit:]
    now = datetime.now()

    lines = []
    for snap in items:
      delta = (now - snap.created_at).total_seconds()
      if delta < 60:
        time_label = f"约{int(delta)}秒前"
      else:
        time_label = f"约{int(delta / 60)}分钟前"
      lines.append(f"- {time_label}: {snap.description}")

    return "[最近画面变化]\n" + "\n".join(lines) + "\n\n"

  async def stop(self) -> None:
    """取消所有后台任务"""
    for task in self._background_tasks:
      task.cancel()
    if self._background_tasks:
      await asyncio.wait(self._background_tasks, timeout=3.0)
    self._background_tasks.clear()

  def debug_state(self) -> dict:
    """调试状态快照"""
    return {
      "enabled": self._config.enabled,
      "buffer_size": len(self._recent),
      "buffer_capacity": self._config.buffer_size,
      "describing": self._describing,
      "last_describe_sec": round(self._last_describe_sec, 1),
      "recent": [
        {
          "timestamp_sec": round(s.timestamp_sec, 1),
          "description": s.description[:80],
          "age_seconds": round((datetime.now() - s.created_at).total_seconds()),
        }
        for s in self._recent
      ],
    }
