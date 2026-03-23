"""
Controller 桥接层

将 StreamingStudio 的运行时状态映射为 ControllerInput，
并将 PromptPlan 应用到执行层。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

from llm_controller.schema import (
  CommentBrief,
  ControllerInput,
  TopicBrief,
  ViewerBrief,
)

if TYPE_CHECKING:
  from .models import Comment
  from .guard_roster import GuardRoster
  from memory.manager import MemoryManager
  from topic_manager.manager import TopicManager
  from broadcaster_state.state_card import StateCard
  from .scene_memory import SceneMemoryCache


_GUARD_LEVEL_NAMES = {1: "舰长", 2: "提督", 3: "总督"}


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


def build_controller_input(
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
  available_persona_sections: tuple[str, ...] = (),
  available_knowledge_topics: tuple[str, ...] = (),
  available_corpus_styles: tuple[str, ...] = (),
  available_corpus_scenes: tuple[str, ...] = (),
) -> ControllerInput:
  """从 Studio 运行时状态构建 ControllerInput"""
  now = datetime.now()

  comment_briefs: list[CommentBrief] = []
  viewer_ids_seen: set[str] = set()

  all_comments = old_comments + new_comments
  new_ids = {c.id for c in new_comments}
  viewer_nickname_map = {
    c.user_id: str(c.nickname).strip()
    for c in all_comments
    if str(c.nickname).strip()
  }

  for c in all_comments:
    seconds_ago = (now - c.timestamp).total_seconds()
    member = guard_roster.get_member_by_nickname(c.nickname)
    is_guard = member is not None
    guard_level_name = member.level_name if member else ""

    comment_briefs.append(CommentBrief(
      id=c.id,
      user_id=c.user_id,
      nickname=c.nickname,
      content=_controller_content(c),
      event_type=c.event_type.value if hasattr(c.event_type, 'value') else str(c.event_type),
      price=getattr(c, 'price', 0.0) or 0.0,
      guard_level=getattr(c, 'guard_level', 0) or 0,
      is_guard_member=is_guard,
      guard_member_level=guard_level_name,
      seconds_ago=seconds_ago,
      is_new=c.id in new_ids,
    ))
    viewer_ids_seen.add(c.user_id)

  viewer_briefs: list[ViewerBrief] = []
  if memory_manager is not None:
    user_store = memory_manager.user_memory_store
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
        nickname = str(identity.get("preferred_address", "")).strip() or nickname
        rel = record.relationship_state or {}
        familiarity = float(rel.get("familiarity", 0) or 0)
        trust = float(rel.get("trust", 0) or 0)
        has_callbacks = bool(record.callbacks)
        has_open_threads = bool(record.open_threads)
        if record.topic_profile:
          last_topic_str = str(record.topic_profile[-1].get("topic", ""))

      member = guard_roster.get_member_by_nickname(comment_nickname or nickname)

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
  if topic_manager is not None:
    for t in topic_manager.table.get_all():
      idle = (now - t.last_discussed_at).total_seconds()
      topic_briefs.append(TopicBrief(
        topic_id=t.topic_id,
        title=t.title,
        significance=t.significance,
        stale=t.stale,
        idle_seconds=idle,
      ))

  energy = 0.7
  patience = 0.7
  atmosphere = ""
  emotion = ""
  if state_card is not None:
    energy = state_card.energy
    patience = state_card.patience
    atmosphere = getattr(state_card, 'atmosphere', '') or ""
    emotion = getattr(state_card, 'emotion', '') or ""

  return ControllerInput(
    energy=energy,
    patience=patience,
    atmosphere=atmosphere,
    emotion=emotion,
    stream_phase=stream_phase,
    round_count=round_count,
    comments=tuple(comment_briefs),
    comment_rate=comment_rate,
    silence_seconds=silence_seconds,
    viewer_briefs=tuple(viewer_briefs),
    active_topics=tuple(topic_briefs),
    is_conversation_mode=is_conversation_mode,
    has_scene_change=has_scene_change,
    scene_description=scene_description,
    last_response_style=last_response_style,
    last_topic=last_topic,
    available_persona_sections=available_persona_sections,
    available_knowledge_topics=available_knowledge_topics,
    available_corpus_styles=available_corpus_styles,
    available_corpus_scenes=available_corpus_scenes,
  )
