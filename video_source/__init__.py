"""
视频源模块
从视频文件提取帧、解析 B站弹幕 XML、按时间轴同步回放
"""

from .frame_extractor import FrameExtractor
from .danmaku_parser import DanmakuParser, Danmaku
from .video_player import VideoPlayer

__all__ = [
  "FrameExtractor",
  "DanmakuParser",
  "Danmaku",
  "VideoPlayer",
]
