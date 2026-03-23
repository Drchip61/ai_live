"""
路由化 prompt 组合器
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prompts import PromptLoader

from .models import Comment, EventType, GUARD_LEVEL_NAMES


@dataclass(frozen=True)
class RoutePromptBundle:
  """单轮回复所需的 prompt 组合结果"""

  prompt: str
  reply_images: Optional[list[str]]
  rag_query: str = ""
  memory_input: str = ""


class RoutePromptComposer:
  """按 route_kind 组合用户消息 prompt"""

  def __init__(self, prompt_loader: Optional[PromptLoader] = None):
    self._loader = prompt_loader or PromptLoader()
    self._route_prompts = {
      "chat": self._loader.get_route_instruction("chat"),
      "super_chat": self._loader.get_route_instruction("super_chat"),
      "gift": self._loader.get_route_instruction("gift"),
      "guard_buy": self._loader.get_route_instruction("guard_buy"),
      "entry": self._loader.get_route_instruction("entry"),
      "vlm": self._loader.get_route_instruction("vlm"),
      "proactive": self._loader.get_route_instruction("proactive"),
    }

  def compose(
    self,
    route_kind: str,
    formatted_comments: str,
    old_comments: list[Comment],
    new_comments: list[Comment],
    time_tag: str,
    conversation_mode: bool,
    scene_context: str = "",
    stream_timestamp: str = "",
    images: Optional[list[str]] = None,
    session_mode: str = "none",
    fake_gift_ids: tuple[str, ...] = (),
  ) -> RoutePromptBundle:
    """按路由组合 prompt、图像和记忆查询文本"""
    route = route_kind if route_kind in self._route_prompts else "chat"
    allow_visual = bool(images) and route in ("vlm", "proactive")
    if session_mode == "video_focus" and images:
      allow_visual = True
    if route in ("gift", "guard_buy", "entry"):
      allow_visual = False

    reply_images = images if allow_visual else None
    parts: list[str] = [self._route_prompts[route], time_tag.strip()]

    if reply_images:
      if route != "vlm":
        parts.append(self._route_prompts["vlm"])
      if scene_context:
        parts.append(scene_context.strip())
      if stream_timestamp:
        parts.append(f"[当前画面] 以下附带了直播画面截图（{stream_timestamp}）。")
    elif conversation_mode:
      parts.append("[当前模式] 纯对话模式，没有直播画面，请专注于弹幕互动。")

    fake_gift_block = self._build_fake_gift_block(fake_gift_ids, old_comments + new_comments)
    if fake_gift_block:
      parts.append(fake_gift_block)

    if formatted_comments.strip():
      parts.append(formatted_comments.strip())

    rag_query = self._build_rag_query(route, old_comments, new_comments)
    memory_input = self._build_memory_input(
      route,
      old_comments,
      new_comments,
      scene_context=scene_context,
    )
    prompt = "\n\n".join(part for part in parts if part).strip()
    return RoutePromptBundle(
      prompt=prompt,
      reply_images=reply_images,
      rag_query=rag_query,
      memory_input=memory_input,
    )

  @staticmethod
  def _build_fake_gift_block(
    fake_gift_ids: tuple[str, ...],
    comments: list[Comment],
  ) -> str:
    if not fake_gift_ids:
      return ""
    lines = ["【伪礼物提醒】以下只是普通文字弹幕，不是真实礼物/上舰事件："]
    fake_id_set = set(fake_gift_ids)
    for comment in comments:
      if comment.id not in fake_id_set:
        continue
      text = comment.content.strip()
      if not text:
        continue
      lines.append(f"- {comment.nickname}: {text}")
    return "\n".join(lines) if len(lines) > 1 else ""

  def _build_rag_query(
    self,
    route_kind: str,
    old_comments: list[Comment],
    new_comments: list[Comment],
  ) -> str:
    if route_kind not in ("chat", "super_chat"):
      return ""
    focus = new_comments or old_comments[-3:]
    lines = [
      self._comment_summary(comment)
      for comment in focus[-5:]
      if self._is_meaningful(comment)
    ]
    return " ".join(lines)

  def _build_memory_input(
    self,
    route_kind: str,
    old_comments: list[Comment],
    new_comments: list[Comment],
    scene_context: str = "",
  ) -> str:
    if route_kind in ("gift", "guard_buy", "entry"):
      return ""

    parts: list[str] = []
    focus = new_comments or old_comments[-3:]
    comment_lines = [
      f"观众「{comment.nickname}」：{self._comment_payload(comment)}"
      for comment in focus[-5:]
      if self._is_meaningful(comment)
    ]
    if comment_lines:
      parts.append("；".join(comment_lines))

    if route_kind in ("vlm", "proactive"):
      scene_summary = self._compact_scene_context(scene_context)
      if scene_summary:
        parts.insert(0, scene_summary)

    return "\n".join(part for part in parts if part).strip()

  @staticmethod
  def _compact_scene_context(scene_context: str) -> str:
    if not scene_context.strip():
      return ""
    lines = [line.strip() for line in scene_context.splitlines() if line.strip()]
    return " ".join(lines[:6])

  @staticmethod
  def _is_meaningful(comment: Comment) -> bool:
    payload = RoutePromptComposer._comment_payload(comment)
    return bool(payload.strip()) and len(payload.strip()) > 2

  @staticmethod
  def _comment_summary(comment: Comment) -> str:
    return f"{comment.nickname}：{RoutePromptComposer._comment_payload(comment)}"

  @staticmethod
  def _comment_payload(comment: Comment) -> str:
    if comment.event_type == EventType.GUARD_BUY:
      level = GUARD_LEVEL_NAMES.get(comment.guard_level, "舰长")
      return f"开通了{level}"
    if comment.event_type == EventType.SUPER_CHAT:
      return f"SC ¥{comment.price:.0f}：{comment.content.strip()}"
    if comment.event_type == EventType.GIFT:
      gift_name = comment.gift_name or "礼物"
      return f"赠送 {gift_name} x{comment.gift_num}"
    if comment.event_type == EventType.ENTRY:
      return "进入直播间"
    return comment.content.strip()
