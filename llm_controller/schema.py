"""
Controller 数据结构

ControllerInput  — Controller 的输入（元数据级摘要）
PromptPlan       — Controller 的输出（结构化决策）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# 输入侧
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CommentBrief:
  """单条弹幕/事件的元数据摘要"""
  id: str
  user_id: str
  nickname: str
  content: str
  event_type: str = "danmaku"
  price: float = 0.0
  guard_level: int = 0
  is_guard_member: bool = False
  guard_member_level: str = ""
  seconds_ago: float = 0.0
  is_new: bool = True

  def to_prompt_line(self) -> str:
    tag = "新" if self.is_new else "旧"
    time_str = f"{self.seconds_ago:.0f}s前"
    parts = [f"[{tag}]", f"[{time_str}]", self.nickname]

    event_info: list[str] = []
    if self.event_type != "danmaku":
      event_info.append(self.event_type)
    if self.price > 0:
      event_info.append(f"¥{self.price:.0f}")
    if self.is_guard_member:
      event_info.append(f"会员:{self.guard_member_level}")
    if event_info:
      parts.append(f"({', '.join(event_info)})")

    parts.append(f": {self.content}")
    return " ".join(parts)


@dataclass(frozen=True)
class ViewerBrief:
  """观众元数据摘要"""
  viewer_id: str
  nickname: str
  familiarity: float = 0.0
  trust: float = 0.0
  visit_count: int = 0
  has_callbacks: bool = False
  has_open_threads: bool = False
  last_topic: str = ""
  is_guard_member: bool = False
  guard_level_name: str = ""

  def to_prompt_line(self) -> str:
    parts = [self.nickname]
    if self.is_guard_member:
      parts.append(f"[{self.guard_level_name}]")
    parts.append(f"熟悉度:{self.familiarity:.1f}")
    parts.append(f"信任:{self.trust:.1f}")
    if self.has_open_threads:
      parts.append("有未了话头")
    if self.has_callbacks:
      parts.append("有回钩线索")
    if self.last_topic:
      parts.append(f"上次话题:{self.last_topic}")
    return " | ".join(parts)


@dataclass(frozen=True)
class TopicBrief:
  """话题元数据摘要"""
  topic_id: str
  title: str
  significance: float = 0.5
  stale: bool = False
  idle_seconds: float = 0.0

  def to_prompt_line(self) -> str:
    stale_tag = " [过期]" if self.stale else ""
    idle_tag = f" (空闲{self.idle_seconds:.0f}s)" if self.idle_seconds > 20 else ""
    return f"{self.topic_id}: {self.title} (重要度:{self.significance:.2f}{stale_tag}{idle_tag})"


@dataclass(frozen=True)
class ControllerInput:
  """Controller 单次调度的完整输入"""
  # 主播状态
  energy: float = 0.7
  patience: float = 0.7
  atmosphere: str = ""
  emotion: str = ""
  stream_phase: str = "开场"
  round_count: int = 0

  # 弹幕
  comments: tuple[CommentBrief, ...] = ()
  comment_rate: float = -1.0
  silence_seconds: float = 0.0

  # 观众
  viewer_briefs: tuple[ViewerBrief, ...] = ()

  # 话题
  active_topics: tuple[TopicBrief, ...] = ()

  # 场景
  is_conversation_mode: bool = False
  has_scene_change: bool = False
  scene_description: str = ""

  # 上一轮
  last_response_style: str = "normal"
  last_topic: str = ""

  # 可用资源目录（启动时缓存，传入供 prompt 渲染）
  available_persona_sections: tuple[str, ...] = ()
  available_knowledge_topics: tuple[str, ...] = ()
  available_corpus_styles: tuple[str, ...] = ()
  available_corpus_scenes: tuple[str, ...] = ()

  @property
  def new_comments(self) -> list[CommentBrief]:
    return [c for c in self.comments if c.is_new]

  @property
  def old_comments(self) -> list[CommentBrief]:
    return [c for c in self.comments if not c.is_new]


# ------------------------------------------------------------------
# 输出侧
# ------------------------------------------------------------------

_VALID_STYLES = frozenset({
  "reaction", "brief", "normal", "detailed",
  "existential", "guard_thanks",
})

_VALID_ROUTE_KINDS = frozenset({
  "chat",
  "super_chat",
  "gift",
  "guard_buy",
  "entry",
  "vlm",
  "proactive",
})


@dataclass(frozen=True)
class PromptPlan:
  """Controller 的结构化决策输出"""
  # 回复决策
  should_reply: bool = True
  urgency: int = 5
  route_kind: str = "chat"

  # 风格控制
  response_style: str = "normal"
  sentences: int = 2
  tone_hint: str = ""

  # 记忆选择
  memory_strategy: str = "normal"
  viewer_focus_ids: tuple[str, ...] = ()

  # 角色记忆定向检索
  persona_sections: tuple[str, ...] = ()

  # 语料库定向检索
  corpus_style: str = ""
  corpus_scene: str = ""

  # 外部知识库
  knowledge_topics: tuple[str, ...] = ()

  # 话题分类
  topic_assignments: dict[str, str] = field(default_factory=dict)

  # 假礼物检测
  fake_gift_ids: tuple[str, ...] = ()

  # 会话控制
  session_mode: str = "none"
  session_anchor: str = ""

  # 调度建议
  suggested_wait_min: float = 3.0
  suggested_wait_max: float = 8.0
  priority: int = 1

  # 主动发言
  proactive_speak: bool = False
  proactive_reason: str = ""

  # 动态指令
  extra_instructions: tuple[str, ...] = ()

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> PromptPlan:
    """从 Controller LLM 的 JSON 输出构建 PromptPlan，所有字段有安全默认值"""
    style = data.get("response_style", "normal")
    if style not in _VALID_STYLES:
      style = "normal"

    sentences = data.get("sentences", 2)
    sentences = max(1, min(4, int(sentences))) if sentences is not None else 2

    urgency = data.get("urgency", 5)
    urgency = max(0, min(9, int(urgency))) if urgency is not None else 5

    priority = data.get("priority", 1)
    priority = max(0, min(3, int(priority))) if priority is not None else 1

    route_kind = str(data.get("route_kind", "chat") or "chat").strip()
    if route_kind not in _VALID_ROUTE_KINDS:
      route_kind = "chat"

    memory_strategy = data.get("memory_strategy", "normal")
    if memory_strategy not in ("minimal", "normal", "deep_recall"):
      memory_strategy = "normal"

    session_mode = data.get("session_mode", "none")
    if session_mode not in ("none", "comment_focus", "video_focus"):
      session_mode = "none"

    def _to_str_tuple(val: Any) -> tuple[str, ...]:
      if isinstance(val, (list, tuple)):
        return tuple(str(v) for v in val if v)
      return ()

    return cls(
      should_reply=bool(data.get("should_reply", True)),
      urgency=urgency,
      route_kind=route_kind,
      response_style=style,
      sentences=sentences,
      tone_hint=str(data.get("tone_hint", "") or ""),
      memory_strategy=memory_strategy,
      viewer_focus_ids=_to_str_tuple(data.get("viewer_focus_ids")),
      persona_sections=_to_str_tuple(data.get("persona_sections")),
      corpus_style=str(data.get("corpus_style", "") or ""),
      corpus_scene=str(data.get("corpus_scene", "") or ""),
      knowledge_topics=_to_str_tuple(data.get("knowledge_topics")),
      topic_assignments=data.get("topic_assignments") or {},
      fake_gift_ids=_to_str_tuple(data.get("fake_gift_ids")),
      session_mode=session_mode,
      session_anchor=str(data.get("session_anchor", "") or ""),
      suggested_wait_min=float(data.get("suggested_wait_min", 3.0) or 3.0),
      suggested_wait_max=float(data.get("suggested_wait_max", 8.0) or 8.0),
      priority=priority,
      proactive_speak=bool(data.get("proactive_speak", False)),
      proactive_reason=str(data.get("proactive_reason", "") or ""),
      extra_instructions=_to_str_tuple(data.get("extra_instructions")),
    )
