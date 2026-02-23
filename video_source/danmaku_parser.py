"""
B站弹幕 XML 解析器
解析标准 B站弹幕 XML 格式，提取弹幕内容和时间信息
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class Danmaku:
  """
  一条弹幕数据

  Attributes:
    time_sec: 弹幕出现时间（视频内秒数）
    content: 弹幕文本内容
    user_hash: 用户哈希（B站匿名化后的标识）
    mode: 弹幕类型 (1-3 滚动, 4 底部, 5 顶部, 6 逆向, 7 特殊, 8 高级)
    font_size: 字号
    color: 颜色（十进制整数）
    send_timestamp: 发送时的 UNIX 时间戳
    row_id: 弹幕 ID
  """
  time_sec: float
  content: str
  user_hash: str = ""
  mode: int = 1
  font_size: int = 25
  color: int = 16777215
  send_timestamp: int = 0
  row_id: str = ""


class DanmakuParser:
  """
  B站弹幕 XML 解析器

  标准 B站弹幕 XML 格式：
  ```xml
  <i>
    <d p="time,mode,fontSize,color,timestamp,pool,userId,rowId">弹幕内容</d>
    ...
  </i>
  ```

  p 属性的逗号分隔字段：
    [0] time: 弹幕出现时间（秒，小数）
    [1] mode: 弹幕类型
    [2] fontSize: 字号
    [3] color: 颜色（十进制）
    [4] timestamp: 发送时间戳（UNIX）
    [5] pool: 弹幕池
    [6] userId: 用户哈希
    [7] rowId: 弹幕 ID
  """

  def __init__(self, xml_path: Union[str, Path]):
    """
    Args:
      xml_path: B站弹幕 XML 文件路径
    """
    self.xml_path = Path(xml_path)
    if not self.xml_path.exists():
      raise FileNotFoundError(f"弹幕文件不存在: {self.xml_path}")

    self._danmakus: list[Danmaku] = []
    self._parse()

  def _parse(self) -> None:
    """解析 XML 文件"""
    tree = ET.parse(self.xml_path)
    root = tree.getroot()

    for elem in root.iter("d"):
      p_attr = elem.get("p", "")
      text = elem.text or ""
      text = text.strip()

      if not text or not p_attr:
        continue

      parts = p_attr.split(",")
      if len(parts) < 8:
        continue

      try:
        dm = Danmaku(
          time_sec=float(parts[0]),
          content=text,
          mode=int(parts[1]),
          font_size=int(parts[2]),
          color=int(parts[3]),
          send_timestamp=int(parts[4]),
          user_hash=parts[6],
          row_id=parts[7],
        )
        self._danmakus.append(dm)
      except (ValueError, IndexError):
        continue

    self._danmakus.sort(key=lambda d: d.time_sec)

  @property
  def danmakus(self) -> list[Danmaku]:
    """所有弹幕（按时间排序）"""
    return self._danmakus.copy()

  @property
  def count(self) -> int:
    return len(self._danmakus)

  @property
  def duration(self) -> float:
    """弹幕覆盖的时间范围（秒）"""
    if not self._danmakus:
      return 0
    return self._danmakus[-1].time_sec

  def get_range(self, start_sec: float, end_sec: float) -> list[Danmaku]:
    """
    获取指定时间范围内的弹幕

    Args:
      start_sec: 起始时间（秒）
      end_sec: 结束时间（秒）

    Returns:
      该时间段内的弹幕列表
    """
    return [
      d for d in self._danmakus
      if start_sec <= d.time_sec < end_sec
    ]

  def get_before(self, timestamp_sec: float, limit: int = 20) -> list[Danmaku]:
    """
    获取指定时间点之前的最近 N 条弹幕

    Args:
      timestamp_sec: 时间点（秒）
      limit: 最大返回数量

    Returns:
      弹幕列表（按时间升序）
    """
    before = [d for d in self._danmakus if d.time_sec < timestamp_sec]
    return before[-limit:]

  def filter_normal(self) -> list[Danmaku]:
    """过滤出普通滚动弹幕（mode 1-3），排除特殊弹幕"""
    return [d for d in self._danmakus if d.mode in (1, 2, 3)]

  def __repr__(self) -> str:
    return f"DanmakuParser({self.xml_path.name}, {self.count} danmakus, {self.duration:.1f}s)"
