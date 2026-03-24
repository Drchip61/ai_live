"""
数据模型
定义弹幕（含礼物/SC/上舰/入场等事件）和主播回复的数据结构
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EventType(str, Enum):
  """弹幕/事件类型"""
  DANMAKU = "danmaku"
  GIFT = "gift"
  SUPER_CHAT = "super_chat"
  GUARD_BUY = "guard_buy"
  ENTRY = "entry"


GUARD_LEVEL_NAMES = {1: "舰长", 2: "提督", 3: "总督"}

EVENT_PRIORITY_ORDER = {
  EventType.GUARD_BUY: 0,
  EventType.SUPER_CHAT: 1,
  EventType.GIFT: 2,
  EventType.DANMAKU: 3,
  EventType.ENTRY: 4,
}


@dataclass(frozen=True)
class Comment:
  """
  弹幕/事件数据模型

  Attributes:
    user_id: 用户ID
    nickname: 用户昵称
    content: 弹幕内容（礼物/入场事件可为空字符串）
    event_type: 事件类型
    gift_name: 礼物名称（gift/guard_buy）
    gift_num: 礼物数量或月数（gift: 个数, guard_buy: 月数）
    price: 金额（super_chat 的 SC 价格，单位元）
    guard_level: 舰长等级（0=无, 1=舰长, 2=提督, 3=总督）
  """
  user_id: str
  nickname: str
  content: str
  id: str = field(default_factory=lambda: str(uuid.uuid4()))
  timestamp: datetime = field(default_factory=datetime.now)
  received_at: datetime = field(default_factory=datetime.now)
  receive_seq: int = 0
  priority: bool = False
  event_type: EventType = EventType.DANMAKU
  gift_name: str = ""
  gift_num: int = 0
  price: float = 0.0
  guard_level: int = 0

  @property
  def is_paid_event(self) -> bool:
    return self.event_type in (EventType.SUPER_CHAT, EventType.GUARD_BUY)

  def to_dict(self) -> dict:
    """转换为字典"""
    d = {
      "id": self.id,
      "user_id": self.user_id,
      "nickname": self.nickname,
      "content": self.content,
      "timestamp": self.timestamp.isoformat(),
      "received_at": self.received_at.isoformat(),
      "receive_seq": self.receive_seq,
      "event_type": self.event_type.value,
    }
    if self.gift_name:
      d["gift_name"] = self.gift_name
    if self.gift_num:
      d["gift_num"] = self.gift_num
    if self.price:
      d["price"] = self.price
    if self.guard_level:
      d["guard_level"] = self.guard_level
    return d

  @classmethod
  def from_dict(cls, data: dict) -> "Comment":
    """从字典创建实例"""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
      timestamp = datetime.fromisoformat(timestamp)
    elif timestamp is None:
      timestamp = datetime.now()
    received_at = data.get("received_at")
    if isinstance(received_at, str):
      received_at = datetime.fromisoformat(received_at)
    elif received_at is None:
      received_at = timestamp

    event_type_str = data.get("event_type", "danmaku")
    try:
      event_type = EventType(event_type_str)
    except ValueError:
      event_type = EventType.DANMAKU

    return cls(
      id=data.get("id", str(uuid.uuid4())),
      user_id=data["user_id"],
      nickname=data["nickname"],
      content=data.get("content", ""),
      timestamp=timestamp,
      received_at=received_at,
      receive_seq=int(data.get("receive_seq", 0) or 0),
      event_type=event_type,
      gift_name=data.get("gift_name", ""),
      gift_num=data.get("gift_num", 0),
      price=data.get("price", 0.0),
      guard_level=data.get("guard_level", 0),
    )

  def format_for_llm(self) -> str:
    """格式化为 LLM 输入"""
    if self.event_type == EventType.GUARD_BUY:
      level = GUARD_LEVEL_NAMES.get(self.guard_level, "舰长")
      return f"{self.nickname} 开通了{level}！"
    if self.event_type == EventType.SUPER_CHAT:
      return f"{self.nickname} (SC ¥{self.price}): {self.content}"
    if self.event_type == EventType.GIFT:
      return f"{self.nickname} 赠送了 {self.gift_name} x{self.gift_num}"
    if self.event_type == EventType.ENTRY:
      return f"{self.nickname} 进入了直播间"
    return f"用户 {self.nickname} 说: {self.content}"


@dataclass(frozen=True)
class StreamerResponse:
  """
  主播回复数据模型

  Attributes:
    id: 唯一标识符
    content: 回复内容（原始 LLM 输出）
    reply_to: 回复的弹幕ID列表
    timestamp: 回复时间
    mapped_content: 经过表情/动作/语音情绪映射后的文本（标签已替换为固定集名称）
    expression_motion_tags: 每个标签的映射详情列表
  """
  content: str
  reply_to: tuple[str, ...] = field(default_factory=tuple)
  reply_target_text: str = ""
  nickname: str = ""
  id: str = field(default_factory=lambda: str(uuid.uuid4()))
  timestamp: datetime = field(default_factory=datetime.now)
  mapped_content: Optional[str] = None
  expression_motion_tags: tuple[Any, ...] = field(default_factory=tuple)
  response_style: str = "normal"
  controller_trace: Optional[dict[str, Any]] = None
  timing_trace: Optional[dict[str, Any]] = None

  def to_dict(self) -> dict:
    """转换为字典"""
    d: dict[str, Any] = {
      "id": self.id,
      "content": self.content,
      "reply_to": list(self.reply_to),
      "reply_target_text": self.reply_target_text,
      "nickname": self.nickname,
      "timestamp": self.timestamp.isoformat(),
      "response_style": self.response_style,
    }
    if self.mapped_content is not None:
      d["mapped_content"] = self.mapped_content
      d["expression_motion_tags"] = [
        {
          "original_action": t.original_action,
          "original_emotion": t.original_emotion,
          "original_voice_emotion": t.original_voice_emotion,
          "mapped_motion": t.mapped_motion,
          "mapped_expression": t.mapped_expression,
          "mapped_voice_emotion": t.mapped_voice_emotion,
          "motion_score": t.motion_score,
          "expression_score": t.expression_score,
          "voice_emotion_score": t.voice_emotion_score,
        }
        for t in self.expression_motion_tags
      ]
    if self.controller_trace is not None:
      d["controller_trace"] = self.controller_trace
    if self.timing_trace is not None:
      d["timing_trace"] = self.timing_trace
    return d

  @classmethod
  def from_dict(cls, data: dict) -> "StreamerResponse":
    """从字典创建实例"""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
      timestamp = datetime.fromisoformat(timestamp)
    elif timestamp is None:
      timestamp = datetime.now()

    reply_to = data.get("reply_to", [])
    if isinstance(reply_to, list):
      reply_to = tuple(reply_to)

    return cls(
      id=data.get("id", str(uuid.uuid4())),
      content=data["content"],
      reply_to=reply_to,
      reply_target_text=str(data.get("reply_target_text", "") or ""),
      nickname=str(
        data.get("nickname", data.get("reply_target_nickname", ""))
        or ""
      ),
      timestamp=timestamp,
      mapped_content=data.get("mapped_content"),
      response_style=str(data.get("response_style", "normal") or "normal"),
      controller_trace=data.get("controller_trace") if isinstance(data.get("controller_trace"), dict) else None,
      timing_trace=data.get("timing_trace") if isinstance(data.get("timing_trace"), dict) else None,
    )


@dataclass(frozen=True)
class ResponseChunk:
  """
  流式回复的单个片段

  Attributes:
    response_id: 所属回复的 ID
    chunk: 本次新增的文本片段
    accumulated: 截至目前的累积文本
    done: 是否为最后一个片段
  """
  response_id: str
  chunk: str
  accumulated: str
  done: bool = False
