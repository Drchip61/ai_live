"""
视频帧提取器
使用 OpenCV 从视频文件中按时间戳提取帧，编码为 base64 供 VLM 使用
"""

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoFrame:
  """
  视频帧数据

  Attributes:
    timestamp_sec: 帧在视频中的时间位置（秒）
    base64_jpeg: JPEG 编码的 base64 字符串（直接用于 Claude API）
    width: 帧宽度
    height: 帧高度
    captured_at: 提取时的系统时间
  """
  timestamp_sec: float
  base64_jpeg: str
  width: int
  height: int
  captured_at: datetime = field(default_factory=datetime.now)


class FrameExtractor:
  """
  视频帧提取器

  从视频文件中提取帧并编码为 base64 JPEG，
  供 Claude VLM API 的 image content block 使用。
  """

  def __init__(
    self,
    video_path: Union[str, Path],
    max_width: int = 1280,
    jpeg_quality: int = 75,
  ):
    """
    Args:
      video_path: 视频文件路径
      max_width: 最大输出宽度（等比缩放），控制 token 消耗
      jpeg_quality: JPEG 压缩质量 (1-100)
    """
    self.video_path = Path(video_path)
    if not self.video_path.exists():
      raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

    self.max_width = max_width
    self.jpeg_quality = jpeg_quality

    self._cap = cv2.VideoCapture(str(self.video_path))
    if not self._cap.isOpened():
      raise RuntimeError(f"无法打开视频文件: {self.video_path}")

    self._fps = self._cap.get(cv2.CAP_PROP_FPS)
    self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self._duration = self._frame_count / self._fps if self._fps > 0 else 0
    self._orig_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    self._orig_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  @property
  def fps(self) -> float:
    return self._fps

  @property
  def duration(self) -> float:
    """视频总时长（秒）"""
    return self._duration

  @property
  def frame_count(self) -> int:
    return self._frame_count

  @property
  def resolution(self) -> tuple[int, int]:
    """原始分辨率 (width, height)"""
    return (self._orig_width, self._orig_height)

  def extract_at(self, timestamp_sec: float) -> Optional[VideoFrame]:
    """
    提取指定时间点的帧

    Args:
      timestamp_sec: 时间位置（秒）

    Returns:
      VideoFrame 对象，超出范围返回 None
    """
    if timestamp_sec < 0 or timestamp_sec > self._duration:
      return None

    self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = self._cap.read()
    if not ret:
      return None

    return self._encode_frame(frame, timestamp_sec)

  def extract_range(
    self,
    start_sec: float,
    end_sec: float,
    interval_sec: float = 5.0,
  ) -> list[VideoFrame]:
    """
    提取时间范围内的帧序列

    Args:
      start_sec: 起始时间（秒）
      end_sec: 结束时间（秒）
      interval_sec: 采样间隔（秒）

    Returns:
      VideoFrame 列表
    """
    frames = []
    t = start_sec
    while t <= end_sec:
      frame = self.extract_at(t)
      if frame:
        frames.append(frame)
      t += interval_sec
    return frames

  def _encode_frame(self, frame: np.ndarray, timestamp_sec: float) -> VideoFrame:
    """将 OpenCV 帧缩放并编码为 base64 JPEG"""
    h, w = frame.shape[:2]

    if w > self.max_width:
      scale = self.max_width / w
      new_w = self.max_width
      new_h = int(h * scale)
      frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
      h, w = new_h, new_w

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
    _, buf = cv2.imencode(".jpg", frame, encode_params)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return VideoFrame(
      timestamp_sec=timestamp_sec,
      base64_jpeg=b64,
      width=w,
      height=h,
    )

  def close(self):
    """释放视频资源"""
    if self._cap and self._cap.isOpened():
      self._cap.release()

  def __del__(self):
    self.close()

  def __repr__(self) -> str:
    return (
      f"FrameExtractor({self.video_path.name}, "
      f"{self._orig_width}x{self._orig_height}, "
      f"{self._duration:.1f}s, {self._fps:.1f}fps)"
    )
