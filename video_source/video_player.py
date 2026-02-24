"""
视频时间轴播放器
按真实时间（或倍速）同步推进视频帧和弹幕，供 StreamingStudio 消费
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from .frame_extractor import FrameExtractor, VideoFrame
from .danmaku_parser import DanmakuParser, Danmaku

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackSnapshot:
  """
  当前播放位置的快照

  Attributes:
    current_sec: 当前播放位置（视频秒数）
    frame: 最近一帧画面
    recent_danmakus: 自上次快照以来的新弹幕列表
    is_finished: 视频是否播完
  """
  current_sec: float
  frame: Optional[VideoFrame]
  recent_danmakus: list[Danmaku] = field(default_factory=list)
  is_finished: bool = False


class VideoPlayer:
  """
  异步视频时间轴播放器

  模拟视频播放，按真实时间推进，同步产出视频帧和弹幕。
  支持两种帧率：高帧率显示帧（流畅播放）和低帧率 AI 采样帧（节省 token）。

  用法:
    player = VideoPlayer(video_path, danmaku_path)
    await player.start()
    ...
    snapshot = player.get_snapshot()  # 获取当前时间点的帧+弹幕
    ...
    await player.stop()
  """

  def __init__(
    self,
    video_path: str,
    danmaku_path: Optional[str] = None,
    speed: float = 1.0,
    frame_interval: float = 5.0,
    max_width: int = 1280,
    jpeg_quality: int = 75,
    display_fps: float = 10.0,
    display_max_width: int = 960,
    display_jpeg_quality: int = 50,
  ):
    """
    Args:
      video_path: 视频文件路径
      danmaku_path: B站弹幕 XML 文件路径（可选）
      speed: 播放速度倍率（1.0 = 实时）
      frame_interval: AI 帧采样间隔（秒，视频时间）
      max_width: AI 帧最大宽度
      jpeg_quality: AI 帧 JPEG 压缩质量
      display_fps: 显示帧率（仅用于 GUI 流畅播放，不影响 AI）
      display_max_width: 显示帧最大宽度（可比 AI 帧低以提高速度）
      display_jpeg_quality: 显示帧 JPEG 压缩质量
    """
    self.speed = speed
    self.frame_interval = frame_interval
    self.display_fps = display_fps

    # AI 帧提取器（高质量，低频率）
    self._extractor = FrameExtractor(
      video_path,
      max_width=max_width,
      jpeg_quality=jpeg_quality,
    )

    # 显示帧提取器（低质量，高频率，独立 VideoCapture 避免 seek 冲突）
    self._display_extractor: Optional[FrameExtractor] = None
    if display_fps > 0:
      self._display_extractor = FrameExtractor(
        video_path,
        max_width=display_max_width,
        jpeg_quality=display_jpeg_quality,
      )

    self._parser = DanmakuParser(danmaku_path) if danmaku_path else None

    # 播放状态
    self._running = False
    self._start_time: Optional[datetime] = None
    self._current_sec: float = 0.0
    self._paused = False

    # 最新帧缓存（AI 帧）
    self._current_frame: Optional[VideoFrame] = None
    self._last_frame_sec: float = -999.0

    # 显示帧状态
    self._last_display_frame_sec: float = -999.0
    self._display_interval: float = 1.0 / display_fps if display_fps > 0 else 999.0

    # 弹幕已消费位置
    self._danmaku_cursor: float = 0.0
    # 自上次 get_snapshot 以来积累的新弹幕
    self._pending_danmakus: deque[Danmaku] = deque(maxlen=200)

    # 后台任务
    self._tick_task: Optional[asyncio.Task] = None

    # 事件回调
    self._on_danmaku_callbacks: list[Callable[[Danmaku], None]] = []
    self._on_frame_callbacks: list[Callable[[VideoFrame], None]] = []
    self._on_display_frame_callbacks: list[Callable[[VideoFrame], None]] = []

  @property
  def duration(self) -> float:
    """视频总时长（秒）"""
    return self._extractor.duration

  @property
  def current_sec(self) -> float:
    """当前播放位置（视频秒数）"""
    return self._current_sec

  @property
  def is_running(self) -> bool:
    return self._running

  @property
  def is_finished(self) -> bool:
    return self._current_sec >= self._extractor.duration

  @property
  def current_frame(self) -> Optional[VideoFrame]:
    """最新一帧"""
    return self._current_frame

  def on_danmaku(self, callback: Callable[[Danmaku], None]) -> None:
    """注册弹幕到达回调"""
    self._on_danmaku_callbacks.append(callback)

  def on_frame(self, callback: Callable[[VideoFrame], None]) -> None:
    """注册 AI 采样帧回调（低频率，每 frame_interval 秒一次）"""
    self._on_frame_callbacks.append(callback)

  def on_display_frame(self, callback: Callable[[VideoFrame], None]) -> None:
    """注册显示帧回调（高频率，用于 GUI 流畅播放）"""
    self._on_display_frame_callbacks.append(callback)

  async def start(self) -> None:
    """开始播放"""
    if self._running:
      return
    self._running = True
    self._start_time = datetime.now()
    self._current_sec = 0.0
    self._danmaku_cursor = 0.0
    self._tick_task = asyncio.create_task(self._tick_loop())
    logger.info(
      "VideoPlayer 启动: %s, 时长 %.1fs, 速度 %.1fx",
      self._extractor.video_path.name,
      self.duration,
      self.speed,
    )

  async def stop(self) -> None:
    """停止播放"""
    self._running = False
    if self._tick_task:
      self._tick_task.cancel()
      try:
        await self._tick_task
      except asyncio.CancelledError:
        pass
      self._tick_task = None
    self._extractor.close()
    if self._display_extractor:
      self._display_extractor.close()

  def get_snapshot(self) -> PlaybackSnapshot:
    """
    获取当前播放位置的快照（帧 + 累积弹幕）

    调用后清空弹幕缓冲，下次调用只返回增量弹幕。
    """
    danmakus = list(self._pending_danmakus)
    self._pending_danmakus.clear()

    return PlaybackSnapshot(
      current_sec=self._current_sec,
      frame=self._current_frame,
      recent_danmakus=danmakus,
      is_finished=self.is_finished,
    )

  async def _tick_loop(self) -> None:
    """
    后台时钟循环：高频驱动显示帧 + 低频驱动 AI 帧 + 弹幕投递

    tick_interval 取决于是否有显示帧回调：有则 0.05s（~20fps 上限），否则 0.5s
    """
    has_display = bool(self._on_display_frame_callbacks and self._display_extractor)
    tick_interval = 0.05 if has_display else 0.5

    while self._running:
      try:
        if self._paused:
          await asyncio.sleep(tick_interval)
          continue

        # 推进播放位置
        if self._start_time:
          elapsed = (datetime.now() - self._start_time).total_seconds()
          self._current_sec = elapsed * self.speed

        if self._current_sec >= self._extractor.duration:
          self._running = False
          logger.info("视频播放完毕 (%.1fs)", self._extractor.duration)
          break

        # 显示帧（高频率，用于 GUI 流畅播放）
        if has_display:
          if self._current_sec - self._last_display_frame_sec >= self._display_interval:
            display_frame = self._display_extractor.extract_at(self._current_sec)
            if display_frame:
              self._last_display_frame_sec = self._current_sec
              for cb in self._on_display_frame_callbacks:
                try:
                  cb(display_frame)
                except Exception as e:
                  logger.error("显示帧回调错误: %s", e)

        # AI 采样帧（低频率，供 VLM 模型使用）
        if self._current_sec - self._last_frame_sec >= self.frame_interval:
          frame = self._extractor.extract_at(self._current_sec)
          if frame:
            self._current_frame = frame
            self._last_frame_sec = self._current_sec
            for cb in self._on_frame_callbacks:
              try:
                cb(frame)
              except Exception as e:
                logger.error("帧回调错误: %s", e)

        # 投递区间内的弹幕
        if self._parser:
          new_danmakus = self._parser.get_range(
            self._danmaku_cursor, self._current_sec,
          )
          for dm in new_danmakus:
            self._pending_danmakus.append(dm)
            for cb in self._on_danmaku_callbacks:
              try:
                cb(dm)
              except Exception as e:
                logger.error("弹幕回调错误: %s", e)
          self._danmaku_cursor = self._current_sec

        await asyncio.sleep(tick_interval)

      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error("VideoPlayer tick 错误: %s", e)
        await asyncio.sleep(1)

  def debug_state(self) -> dict:
    """调试状态快照"""
    return {
      "is_running": self._running,
      "current_sec": round(self._current_sec, 1),
      "duration": round(self.duration, 1),
      "progress": f"{self._current_sec / self.duration * 100:.1f}%" if self.duration > 0 else "N/A",
      "speed": self.speed,
      "frame_interval": self.frame_interval,
      "display_fps": self.display_fps,
      "has_frame": self._current_frame is not None,
      "pending_danmakus": len(self._pending_danmakus),
      "video": repr(self._extractor),
    }
