"""
数据模型
定义弹幕和主播回复的数据结构
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class Comment:
  """
  弹幕数据模型

  Attributes:
    id: 唯一标识符
    user_id: 用户ID
    nickname: 用户昵称
    content: 弹幕内容
    timestamp: 发送时间
  """
  user_id: str
  nickname: str
  content: str
  id: str = field(default_factory=lambda: str(uuid.uuid4()))
  timestamp: datetime = field(default_factory=datetime.now)
  priority: bool = False

  def to_dict(self) -> dict:
    """转换为字典"""
    return {
      "id": self.id,
      "user_id": self.user_id,
      "nickname": self.nickname,
      "content": self.content,
      "timestamp": self.timestamp.isoformat()
    }

  @classmethod
  def from_dict(cls, data: dict) -> "Comment":
    """从字典创建实例"""
    timestamp = data.get("timestamp")
    if isinstance(timestamp, str):
      timestamp = datetime.fromisoformat(timestamp)
    elif timestamp is None:
      timestamp = datetime.now()

    return cls(
      id=data.get("id", str(uuid.uuid4())),
      user_id=data["user_id"],
      nickname=data["nickname"],
      content=data["content"],
      timestamp=timestamp
    )

  def format_for_llm(self) -> str:
    """格式化为 LLM 输入"""
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
    mapped_content: 经过表情/动作映射后的文本（标签已替换为固定集名称）
    expression_motion_tags: 每个标签的映射详情列表
  """
  content: str
  reply_to: tuple[str, ...] = field(default_factory=tuple)
  id: str = field(default_factory=lambda: str(uuid.uuid4()))
  timestamp: datetime = field(default_factory=datetime.now)
  mapped_content: Optional[str] = None
  expression_motion_tags: tuple[Any, ...] = field(default_factory=tuple)

  def to_dict(self) -> dict:
    """转换为字典"""
    d: dict[str, Any] = {
      "id": self.id,
      "content": self.content,
      "reply_to": list(self.reply_to),
      "timestamp": self.timestamp.isoformat(),
    }
    if self.mapped_content is not None:
      d["mapped_content"] = self.mapped_content
      d["expression_motion_tags"] = [
        {
          "original_action": t.original_action,
          "original_emotion": t.original_emotion,
          "mapped_motion": t.mapped_motion,
          "mapped_expression": t.mapped_expression,
          "motion_score": t.motion_score,
          "expression_score": t.expression_score,
        }
        for t in self.expression_motion_tags
      ]
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
      timestamp=timestamp
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
