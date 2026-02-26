"""
话题数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Topic:
  """
  话题数据模型

  Attributes:
    topic_id: snake_case 描述性 ID（如 "greeting", "game_discussion"）
    significance: 重要性 0~1
    topic_progress: 自然语言描述，表明这个话题现在聊到哪了
    comment_ids: 关联弹幕 ID（按时间排列）
    user_ids: 关联用户 ID（按时间排列，重复用户刷新位置）
    suggestion: 第三方顾问视角的跟进建议
    stale: 是否被标记为"聊太久"
    created_at: 创建时间
    updated_at: 最后更新时间
  """
  topic_id: str
  title: str
  significance: float
  topic_progress: str
  comment_ids: tuple[str, ...] = field(default_factory=tuple)
  user_ids: tuple[str, ...] = field(default_factory=tuple)
  suggestion: str = ""
  stale: bool = False
  created_at: datetime = field(default_factory=datetime.now)
  updated_at: datetime = field(default_factory=datetime.now)
  last_discussed_at: datetime = field(default_factory=datetime.now)
  """最近一次有弹幕关联到此话题的时间（区别于 updated_at：任何字段变更都会更新）"""


@dataclass(frozen=True)
class ContentAnalysisDelta:
  """
  内容分析任务返回的 delta

  Attributes:
    progress_updates: topic_id → 新的 topic_progress
    new_topics: 要创建的新话题列表
    suggestion_updates: topic_id → 新的 suggestion
  """
  progress_updates: dict[str, str] = field(default_factory=dict)
  new_topics: list[Topic] = field(default_factory=list)
  suggestion_updates: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RhythmAnalysisDelta:
  """
  节奏分析任务返回的 delta

  Attributes:
    stale_topic_ids: 被标记为"聊太久"的话题 ID 列表
    suggested_timing: 建议的下次等待时间 (min, max)，None 表示无建议
  """
  stale_topic_ids: list[str] = field(default_factory=list)
  suggested_timing: tuple[float, float] | None = None
