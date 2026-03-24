"""
Controller 桥接层

将 StreamingStudio 的运行时状态先收束为 TurnSnapshot，
再映射为 ControllerInput，稳定 controller / retriever / composer 的输入边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from llm_controller.schema import (
  CommentBrief,
  ControllerInput,
  ResourceCatalog,
  TopicBrief,
  ViewerBrief,
)
from memory.context_schema import resolve_preferred_address

if TYPE_CHECKING:
  from .models import Comment
  from .guard_roster import GuardRoster
  from memory.manager import MemoryManager
  from topic_manager.manager import TopicManager
  from broadcaster_state.state_card import StateCard
  from .scene_memory import SceneMemoryCache


_GUARD_LEVEL_NAMES = {1: "舰长", 2: "提督", 3: "总督"}


@dataclass(frozen=True)
class TurnSnapshot:
  """单轮共享快照，供 controller / retriever / composer 共用。"""

  old_comments: tuple["Comment", ...] = ()
  new_comments: tuple["Comment", ...] = ()
  guard_roster: Optional["GuardRoster"] = None
  memory_manager: Optional["MemoryManager"] = None
  topic_manager: Optional["TopicManager"] = None
  state_card: Optional["StateCard"] = None
  scene_memory: Optional["SceneMemoryCache"] = None
  is_conversation_mode: bool = False
  has_scene_change: bool = False
  scene_description: str = ""
  silence_seconds: float = 0.0
  comment_rate: float = -1.0
  round_count: int = 0
  last_response_style: str = "normal"
  last_topic: str = ""
  stream_phase: str = "直播中"
  resource_catalog: ResourceCatalog = field(default_factory=ResourceCatalog)

  @property
  def all_comments(self) -> list["Comment"]:
    return list(self.old_comments + self.new_comments)


def _controller_content(comment: "Comment") -> str:
  """把运行时 Comment 规范化为适合 controller 判断的文本"""
  event_type = getattr(comment, "event_type", None)
  if hasattr(event_type, "value"):
    event_type = event_type.value

  if event_type == "guard_buy":
    level = _GUARD_LEVEL_NAMES.get(getattr(comment, "guard_level", 0) or 0, "舰长")
    months = getattr(comment, "gift_num", 0) or 0
    month_text = f"，{months}个月" if months else ""
    return f"{comment.nickname} 开通了{level}{month_text}"
  if event_type == "super_chat":
    return f"SC ¥{getattr(comment, 'price', 0.0) or 0.0:.0f}: {comment.content}"
  if event_type == "gift":
    gift_name = getattr(comment, "gift_name", "") or "礼物"
    gift_num = getattr(comment, "gift_num", 0) or 1
    return f"{comment.nickname} 赠送 {gift_name} x{gift_num}"
  if event_type == "entry":
    return f"{comment.nickname} 进入直播间"
  return comment.content


def build_turn_snapshot(
  *,
  old_comments: list["Comment"],
  new_comments: list["Comment"],
  guard_roster: "GuardRoster",
  memory_manager: Optional["MemoryManager"],
  topic_manager: Optional["TopicManager"],
  state_card: Optional["StateCard"],
  scene_memory: Optional["SceneMemoryCache"],
  is_conversation_mode: bool,
  has_scene_change: bool,
  scene_description: str,
  silence_seconds: float,
  comment_rate: float,
  round_count: int,
  last_response_style: str,
  last_topic: str,
  stream_phase: str = "直播中",
  resource_catalog: Optional[ResourceCatalog] = None,
  available_persona_sections: tuple[str, ...] = (),
  available_knowledge_topics: tuple[str, ...] = (),
  available_corpus_styles: tuple[str, ...] = (),
  available_corpus_scenes: tuple[str, ...] = (),
) -> TurnSnapshot:
  catalog = resource_catalog or ResourceCatalog(
    persona_sections=available_persona_sections,
    knowledge_topics=available_knowledge_topics,
    corpus_styles=available_corpus_styles,
    corpus_scenes=available_corpus_scenes,
  )
  resolved_stream_phase = stream_phase
  if state_card is not None and getattr(state_card, "stream_phase", ""):
    resolved_stream_phase = getattr(state_card, "stream_phase", "") or stream_phase

  return TurnSnapshot(
    old_comments=tuple(old_comments),
    new_comments=tuple(new_comments),
    guard_roster=guard_roster,
    memory_manager=memory_manager,
    topic_manager=topic_manager,
    state_card=state_card,
    scene_memory=scene_memory,
    is_conversation_mode=is_conversation_mode,
    has_scene_change=has_scene_change,
    scene_description=scene_description,
    silence_seconds=silence_seconds,
    comment_rate=comment_rate,
    round_count=round_count,
    last_response_style=last_response_style,
    last_topic=last_topic,
    stream_phase=resolved_stream_phase,
    resource_catalog=catalog,
  )


def build_controller_input(
  *,
  snapshot: Optional[TurnSnapshot] = None,
  old_comments: Optional[list["Comment"]] = None,
  new_comments: Optional[list["Comment"]] = None,
  guard_roster: Optional["GuardRoster"] = None,
  memory_manager: Optional["MemoryManager"] = None,
  topic_manager: Optional["TopicManager"] = None,
  state_card: Optional["StateCard"] = None,
  scene_memory: Optional["SceneMemoryCache"] = None,
  is_conversation_mode: bool = False,
  has_scene_change: bool = False,
  scene_description: str = "",
  silence_seconds: float = 0.0,
  comment_rate: float = -1.0,
  round_count: int = 0,
  last_response_style: str = "normal",
  last_topic: str = "",
  stream_phase: str = "直播中",
  resource_catalog: Optional[ResourceCatalog] = None,
  available_persona_sections: tuple[str, ...] = (),
  available_knowledge_topics: tuple[str, ...] = (),
  available_corpus_styles: tuple[str, ...] = (),
  available_corpus_scenes: tuple[str, ...] = (),
) -> ControllerInput:
  """从 Studio 运行时状态构建 ControllerInput。"""
  if snapshot is None:
    if guard_roster is None:
      raise ValueError("build_controller_input 需要 snapshot 或 guard_roster")
    snapshot = build_turn_snapshot(
      old_comments=old_comments or [],
      new_comments=new_comments or [],
      guard_roster=guard_roster,
      memory_manager=memory_manager,
      topic_manager=topic_manager,
      state_card=state_card,
      scene_memory=scene_memory,
      is_conversation_mode=is_conversation_mode,
      has_scene_change=has_scene_change,
      scene_description=scene_description,
      silence_seconds=silence_seconds,
      comment_rate=comment_rate,
      round_count=round_count,
      last_response_style=last_response_style,
      last_topic=last_topic,
      stream_phase=stream_phase,
      resource_catalog=resource_catalog,
      available_persona_sections=available_persona_sections,
      available_knowledge_topics=available_knowledge_topics,
      available_corpus_styles=available_corpus_styles,
      available_corpus_scenes=available_corpus_scenes,
    )

  now = datetime.now()
  comment_briefs: list[CommentBrief] = []
  viewer_ids_seen: set[str] = set()

  all_comments = snapshot.all_comments
  new_ids = {c.id for c in snapshot.new_comments}
  viewer_nickname_map = {
    c.user_id: str(c.nickname).strip()
    for c in all_comments
    if str(c.nickname).strip()
  }

  for comment in all_comments:
    seconds_ago = (now - comment.timestamp).total_seconds()
    member = (
      snapshot.guard_roster.get_member_by_nickname(comment.nickname)
      if snapshot.guard_roster is not None else None
    )
    is_guard = member is not None
    guard_level_name = member.level_name if member else ""

    comment_briefs.append(CommentBrief(
      id=comment.id,
      user_id=comment.user_id,
      nickname=comment.nickname,
      content=_controller_content(comment),
      event_type=comment.event_type.value if hasattr(comment.event_type, "value") else str(comment.event_type),
      price=getattr(comment, "price", 0.0) or 0.0,
      guard_level=getattr(comment, "guard_level", 0) or 0,
      is_guard_member=is_guard,
      guard_member_level=guard_level_name,
      seconds_ago=seconds_ago,
      is_new=comment.id in new_ids,
    ))
    viewer_ids_seen.add(comment.user_id)

  viewer_briefs: list[ViewerBrief] = []
  if snapshot.memory_manager is not None:
    user_store = snapshot.memory_manager.user_memory_store
    for viewer_id in viewer_ids_seen:
      record = user_store.get(viewer_id) if user_store else None
      comment_nickname = viewer_nickname_map.get(viewer_id, "")
      nickname = comment_nickname or viewer_id
      familiarity = 0.0
      trust = 0.0
      has_callbacks = False
      has_open_threads = False
      last_topic_str = ""

      if record is not None:
        identity = record.identity or {}
        nickname = resolve_preferred_address(
          identity,
          fallback_nicknames=tuple(identity.get("nicknames", ())),
          raw_aliases=(viewer_id, comment_nickname),
          fallback=nickname,
        ) or nickname
        rel = record.relationship_state or {}
        familiarity = float(rel.get("familiarity", 0) or 0)
        trust = float(rel.get("trust", 0) or 0)
        has_callbacks = bool(record.callbacks)
        has_open_threads = bool(record.open_threads)
        if record.topic_profile:
          last_topic_str = str(record.topic_profile[-1].get("topic", ""))

      member = (
        snapshot.guard_roster.get_member_by_nickname(comment_nickname or nickname)
        if snapshot.guard_roster is not None else None
      )

      viewer_briefs.append(ViewerBrief(
        viewer_id=viewer_id,
        nickname=nickname,
        familiarity=familiarity,
        trust=trust,
        has_callbacks=has_callbacks,
        has_open_threads=has_open_threads,
        last_topic=last_topic_str,
        is_guard_member=member is not None,
        guard_level_name=member.level_name if member else "",
      ))

  topic_briefs: list[TopicBrief] = []
  if snapshot.topic_manager is not None:
    for topic in snapshot.topic_manager.table.get_all():
      idle = (now - topic.last_discussed_at).total_seconds()
      topic_briefs.append(TopicBrief(
        topic_id=topic.topic_id,
        title=topic.title,
        significance=topic.significance,
        stale=topic.stale,
        idle_seconds=idle,
      ))

  energy = 0.7
  patience = 0.7
  atmosphere = ""
  emotion = ""
  stream_phase_value = snapshot.stream_phase
  if snapshot.state_card is not None:
    energy = snapshot.state_card.energy
    patience = snapshot.state_card.patience
    atmosphere = getattr(snapshot.state_card, "atmosphere", "") or ""
    emotion = (
      getattr(snapshot.state_card, "undigested_emotion", "") or
      getattr(snapshot.state_card, "emotion", "") or
      ""
    )
    stream_phase_value = getattr(snapshot.state_card, "stream_phase", "") or stream_phase_value

  return ControllerInput(
    energy=energy,
    patience=patience,
    atmosphere=atmosphere,
    emotion=emotion,
    stream_phase=stream_phase_value,
    round_count=snapshot.round_count,
    comments=tuple(comment_briefs),
    comment_rate=snapshot.comment_rate,
    silence_seconds=snapshot.silence_seconds,
    viewer_briefs=tuple(viewer_briefs),
    active_topics=tuple(topic_briefs),
    is_conversation_mode=snapshot.is_conversation_mode,
    has_scene_change=snapshot.has_scene_change,
    scene_description=snapshot.scene_description,
    last_response_style=snapshot.last_response_style,
    last_topic=snapshot.last_topic,
    available_persona_sections=snapshot.resource_catalog.persona_sections,
    available_knowledge_topics=snapshot.resource_catalog.knowledge_topics,
    available_corpus_styles=snapshot.resource_catalog.corpus_styles,
    available_corpus_scenes=snapshot.resource_catalog.corpus_scenes,
  )
