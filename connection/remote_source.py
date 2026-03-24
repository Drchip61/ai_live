"""
远程数据源
以 Pull 模式定时从上游服务器拉取截图；弹幕轮询仅保留兼容路径。
实现与 VideoPlayer 相同的回调接口，可直接传入 StreamingStudio(video_player=...)。
"""

import asyncio
import base64
import logging
from datetime import datetime
from typing import Callable, Optional

import aiohttp
import cv2
import numpy as np

from video_source.frame_extractor import VideoFrame
from video_source.danmaku_parser import Danmaku
from streaming_studio.models import Comment
from .danmaku_push_host import (
  RemoteDanmakuItem,
  item_fingerprint,
  parse_snapshot_payload,
  remote_item_to_comment,
)

logger = logging.getLogger(__name__)


class RemoteSource:
  """
  远程数据源（替代 VideoPlayer）

  定时轮询上游服务器的截图和弹幕接口，
  通过回调注入 StreamingStudio 的帧缓存和弹幕缓冲区。

  接口与 VideoPlayer 对齐（duck typing），可直接用于:
    StreamingStudio(video_player=remote_source)
  """

  def __init__(
    self,
    screenshot_url: str = "http://10.81.7.114:8000/screenshot",
    danmaku_url: Optional[str] = None,
    frame_interval: float = 5.0,
    danmaku_interval: float = 1.0,
    max_width: int = 1280,
    jpeg_quality: int = 75,
    http_timeout: float = 10.0,
  ):
    """
    Args:
      screenshot_url: 截图接口 URL（GET → image/jpeg）
      danmaku_url: 弹幕接口 URL（GET → JSON，可选，未指定则不拉取弹幕）
      frame_interval: 截图轮询间隔（秒）
      danmaku_interval: 弹幕轮询间隔（秒）
      max_width: 缩放后最大宽度（节省 VLM token）
      jpeg_quality: JPEG 压缩质量
      http_timeout: HTTP 请求超时（秒）
    """
    self._screenshot_url = screenshot_url
    self._danmaku_url = danmaku_url
    self.frame_interval = frame_interval
    self._danmaku_interval = danmaku_interval
    self._max_width = max_width
    self._jpeg_quality = jpeg_quality
    self._http_timeout = http_timeout

    # 运行状态
    self._running = False
    self._paused = False
    self._session: Optional[aiohttp.ClientSession] = None
    self._frame_task: Optional[asyncio.Task] = None
    self._danmaku_task: Optional[asyncio.Task] = None

    # 弹幕增量游标（上次拉取的 server_time）
    self._danmaku_cursor: float = 0.0
    self._danmaku_error_logged: bool = False
    # 客户端去重：记录已见弹幕指纹（应对上游不支持 after 过滤的情况）
    self._seen_danmaku: set[tuple[str, str, float]] = set()
    self._seen_danmaku_maxlen: int = 500

    # 最新帧缓存
    self._current_frame: Optional[VideoFrame] = None
    self._frame_count: int = 0
    self._danmaku_count: int = 0
    self._last_error: Optional[str] = None

    # 启动时间
    self._start_time: Optional[datetime] = None

    # 回调（与 VideoPlayer 接口对齐）
    self._on_danmaku_callbacks: list[Callable[[Danmaku], None]] = []
    self._on_frame_callbacks: list[Callable[[VideoFrame], None]] = []
    # 富事件回调（直接传 Comment，包含事件类型、礼物、SC 等元数据）
    self._on_comment_callbacks: list[Callable[[Comment], None]] = []

  # ── VideoPlayer 兼容属性 ──

  @property
  def is_running(self) -> bool:
    return self._running

  @property
  def is_finished(self) -> bool:
    return False

  @property
  def is_paused(self) -> bool:
    return self._paused

  @property
  def current_frame(self) -> Optional[VideoFrame]:
    return self._current_frame

  @property
  def duration(self) -> float:
    return float("inf")

  @property
  def current_sec(self) -> float:
    if self._start_time:
      return (datetime.now() - self._start_time).total_seconds()
    return 0.0

  # ── 回调注册（与 VideoPlayer 接口对齐）──

  def on_danmaku(self, callback: Callable[[Danmaku], None]) -> None:
    self._on_danmaku_callbacks.append(callback)

  def on_frame(self, callback: Callable[[VideoFrame], None]) -> None:
    self._on_frame_callbacks.append(callback)

  def on_comment(self, callback: Callable[[Comment], None]) -> None:
    """注册富事件回调（Comment 含事件类型、礼物、SC 等元数据）"""
    self._on_comment_callbacks.append(callback)

  def on_display_frame(self, callback) -> None:
    """兼容接口，远程模式不产出显示帧"""
    pass

  # ── 生命周期 ──

  async def start(self) -> None:
    if self._running:
      return

    self._running = True
    self._paused = False
    self._start_time = datetime.now()

    timeout = aiohttp.ClientTimeout(total=self._http_timeout)
    self._session = aiohttp.ClientSession(timeout=timeout)

    self._frame_task = asyncio.create_task(self._frame_loop())
    if self._danmaku_url:
      self._danmaku_task = asyncio.create_task(self._danmaku_loop())

    sources = [f"截图: {self._screenshot_url} (每{self.frame_interval}s)"]
    if self._danmaku_url:
      sources.append(f"弹幕: {self._danmaku_url} (每{self._danmaku_interval}s)")
    logger.info("RemoteSource 启动: %s", " | ".join(sources))

  async def stop(self) -> None:
    self._running = False

    for task in [self._frame_task, self._danmaku_task]:
      if task:
        task.cancel()
        try:
          await task
        except asyncio.CancelledError:
          pass

    self._frame_task = None
    self._danmaku_task = None

    if self._session:
      await self._session.close()
      self._session = None

    logger.info(
      "RemoteSource 停止: 共拉取 %d 帧, %d 条弹幕",
      self._frame_count, self._danmaku_count,
    )

  def pause(self) -> None:
    if self._running:
      self._paused = True

  def resume(self) -> None:
    if self._running:
      self._paused = False

  # ── 截图轮询 ──

  async def _frame_loop(self) -> None:
    backoff = self.frame_interval

    while self._running:
      try:
        if self._paused:
          await asyncio.sleep(0.5)
          continue

        frame = await self._fetch_screenshot()
        if frame:
          self._current_frame = frame
          self._frame_count += 1
          self._last_error = None
          backoff = self.frame_interval

          if self._frame_count == 1:
            print(f"[RemoteSource] 首帧获取成功: {frame.width}x{frame.height}")

          for cb in self._on_frame_callbacks:
            try:
              cb(frame)
            except Exception as e:
              logger.error("帧回调错误: %s", e)

          await asyncio.sleep(self.frame_interval)
        else:
          if self._frame_count == 0:
            print(f"[RemoteSource] 截图拉取失败，{backoff:.0f}s 后重试...")
          await asyncio.sleep(min(backoff, 30.0))
          backoff = min(backoff * 2, 30.0)

      except asyncio.CancelledError:
        break
      except Exception as e:
        self._last_error = f"截图轮询错误: {e}"
        logger.error(self._last_error)
        await asyncio.sleep(min(backoff, 30.0))
        backoff = min(backoff * 2, 30.0)

  async def _fetch_screenshot(self) -> Optional[VideoFrame]:
    """
    从上游拉取截图，解码为 VideoFrame

    流程: HTTP GET → raw bytes → OpenCV decode → resize → JPEG encode → base64
    """
    if not self._session:
      return None

    try:
      async with self._session.get(self._screenshot_url) as resp:
        if resp.status != 200:
          logger.warning("截图请求失败: HTTP %d", resp.status)
          return None

        raw_bytes = await resp.read()

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
      logger.warning("截图网络错误: %s", e)
      return None

    return await asyncio.to_thread(self._decode_frame, raw_bytes)

  def _decode_frame(self, raw_bytes: bytes) -> Optional[VideoFrame]:
    """同步解码+缩放+编码（在线程中执行，避免阻塞事件循环）"""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
      logger.warning("截图解码失败")
      return None

    h, w = img.shape[:2]

    if w > self._max_width:
      scale = self._max_width / w
      new_w = self._max_width
      new_h = int(h * scale)
      img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
      h, w = new_h, new_w

    _, buf = cv2.imencode(
      ".jpg", img,
      [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
    )
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return VideoFrame(
      timestamp_sec=self.current_sec,
      base64_jpeg=b64,
      width=w,
      height=h,
    )

  # ── 弹幕轮询 ──

  def _item_to_comment(self, item: RemoteDanmakuItem) -> Comment:
    """将中间表示转换为 Comment（含事件类型元数据）"""
    return remote_item_to_comment(item)

  async def _danmaku_loop(self) -> None:
    backoff = self._danmaku_interval

    while self._running:
      try:
        if self._paused:
          await asyncio.sleep(0.5)
          continue

        items = await self._fetch_danmaku()
        if items is not None:
          self._last_error = None
          backoff = self._danmaku_interval
          if self._danmaku_error_logged:
            print("[RemoteSource] 弹幕接口已恢复")
            self._danmaku_error_logged = False

          for item in items:
            fingerprint = item_fingerprint(item)
            if fingerprint in self._seen_danmaku:
              continue

            self._seen_danmaku.add(fingerprint)
            if len(self._seen_danmaku) > self._seen_danmaku_maxlen:
              to_remove = list(self._seen_danmaku)[:len(self._seen_danmaku) - self._seen_danmaku_maxlen]
              for fp in to_remove:
                self._seen_danmaku.discard(fp)

            self._danmaku_count += 1

            # 富事件回调优先（Comment 含事件类型元数据）
            if self._on_comment_callbacks:
              comment = self._item_to_comment(item)
              for cb in self._on_comment_callbacks:
                try:
                  cb(comment)
                except Exception as e:
                  logger.error("事件回调错误: %s", e)
            else:
              # 降级：仅纯弹幕类型通过旧 Danmaku 回调分发
              if item.event_type == "danmaku":
                dm = Danmaku(
                  time_sec=item.timestamp,
                  content=item.content,
                  user_hash=item.user_id,
                  row_id=item.user_id,
                )
                for cb in self._on_danmaku_callbacks:
                  try:
                    cb(dm)
                  except Exception as e:
                    logger.error("弹幕回调错误: %s", e)

          await asyncio.sleep(self._danmaku_interval)
        else:
          await asyncio.sleep(min(backoff, 30.0))
          backoff = min(backoff * 2, 30.0)

      except asyncio.CancelledError:
        break
      except Exception as e:
        self._last_error = f"弹幕轮询错误: {e}"
        logger.error(self._last_error)
        await asyncio.sleep(min(backoff, 30.0))
        backoff = min(backoff * 2, 30.0)

  async def _fetch_danmaku(self) -> Optional[list[RemoteDanmakuItem]]:
    """
    从上游拉取增量事件

    请求: GET {danmaku_url}?after={cursor}
    响应: {danmakus, notifications, gifts, super_chats, guard_buys, server_time}
    """
    if not self._session or not self._danmaku_url:
      return None

    params = {}
    if self._danmaku_cursor > 0:
      params["after"] = str(self._danmaku_cursor)

    try:
      async with self._session.get(self._danmaku_url, params=params) as resp:
        if resp.status != 200:
          if self._danmaku_count == 0 and not self._danmaku_error_logged:
            print(f"[RemoteSource] 弹幕接口不可用 (HTTP {resp.status})，将自动重试")
            self._danmaku_error_logged = True
          return None

        data = await resp.json()

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
      if not self._danmaku_error_logged:
        print(f"[RemoteSource] 弹幕网络错误: {e}，将自动重试")
        self._danmaku_error_logged = True
      return None
    except Exception as e:
      logger.warning("弹幕响应解析错误: %s", e)
      return None

    server_time = float(data.get("server_time", 0.0) or 0.0)
    if server_time > 0:
      self._danmaku_cursor = server_time

    items, _ = parse_snapshot_payload(data)
    return items

  # ── 调试 ──

  def debug_state(self) -> dict:
    return {
      "type": "RemoteSource",
      "is_running": self._running,
      "is_paused": self._paused,
      "uptime_sec": round(self.current_sec, 1),
      "screenshot_url": self._screenshot_url,
      "danmaku_url": self._danmaku_url or "(未配置)",
      "frame_interval": self.frame_interval,
      "danmaku_interval": self._danmaku_interval,
      "frame_count": self._frame_count,
      "danmaku_count": self._danmaku_count,
      "has_frame": self._current_frame is not None,
      "last_error": self._last_error,
    }
