"""
路由化 prompt 组合器
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from langchain_wrapper.contracts import ModelInvocation, RetrievedContextBundle
from prompts import PromptLoader

from .models import Comment


@dataclass(frozen=True)
class RoutePromptBundle:
  """Route 侧用户消息 prompt 结果。"""

  prompt: str
  reply_images: Optional[list[str]]


@dataclass(frozen=True)
class ComposedPrompt:
  """完整组合结果：用户 prompt + system 侧上下文载荷。"""

  route_bundle: RoutePromptBundle
  invocation: ModelInvocation


class RoutePromptComposer:
  """按 route_kind 组合用户消息 prompt。"""

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
    """只负责 route 侧用户消息，不再混入检索/写回种子。"""
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

    prompt = "\n\n".join(part for part in parts if part).strip()
    return RoutePromptBundle(prompt=prompt, reply_images=reply_images)

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


class PromptComposer:
  """负责把 route prompt + 上下文 bundle 渲染成最终模型调用。"""

  def __init__(
    self,
    route_composer: RoutePromptComposer,
    *,
    style_instructions: Optional[dict[str, str]] = None,
    engaging_question_probability: float = 0.0,
    engaging_question_hint: str = "",
    guard_thanks_reference: str = "",
    gift_thanks_reference: str = "",
    super_chat_reference: str = "",
  ) -> None:
    self._route_composer = route_composer
    self._style_instructions = dict(style_instructions or {})
    self._engaging_question_probability = engaging_question_probability
    self._engaging_question_hint = engaging_question_hint
    self._guard_thanks_reference = guard_thanks_reference
    self._gift_thanks_reference = gift_thanks_reference
    self._super_chat_reference = super_chat_reference

  def compose(
    self,
    *,
    plan,
    formatted_comments: str,
    old_comments: list[Comment],
    new_comments: list[Comment],
    time_tag: str,
    conversation_mode: bool,
    scene_context: str = "",
    stream_timestamp: str = "",
    images: Optional[list[str]] = None,
    topic_context: str = "",
    max_chars: int = 0,
    retrieved_context: Optional[RetrievedContextBundle] = None,
  ) -> ComposedPrompt:
    route_bundle = self._route_composer.compose(
      route_kind=getattr(plan, "route_kind", "chat"),
      formatted_comments=formatted_comments,
      old_comments=old_comments,
      new_comments=new_comments,
      time_tag=time_tag,
      conversation_mode=conversation_mode,
      scene_context=scene_context,
      stream_timestamp=stream_timestamp,
      images=images,
      session_mode=getattr(plan, "session_mode", "none"),
      fake_gift_ids=getattr(plan, "fake_gift_ids", ()),
    )
    prompt = route_bundle.prompt

    style_hint = self._build_style_hint(
      getattr(plan, "route_kind", "chat"),
      getattr(plan, "response_style", "normal"),
      getattr(plan, "sentences", 2),
      max_chars=max_chars,
      extra_instructions=getattr(plan, "extra_instructions", ()),
    )
    if style_hint:
      prompt = style_hint + prompt

    retrieved = retrieved_context or RetrievedContextBundle()
    untrusted_parts = []
    if topic_context.strip():
      untrusted_parts.append(topic_context.strip())
    retrieved_untrusted = retrieved.render_untrusted_text()
    if retrieved_untrusted:
      untrusted_parts.append(retrieved_untrusted)
    invocation = ModelInvocation(
      user_prompt=prompt,
      images=route_bundle.reply_images,
      trusted_context=retrieved.render_trusted_text(),
      untrusted_context="\n\n".join(untrusted_parts),
      response_style=getattr(plan, "response_style", "normal"),
      route_kind=getattr(plan, "route_kind", "chat"),
    )
    return ComposedPrompt(route_bundle=route_bundle, invocation=invocation)

  def _build_style_hint(
    self,
    route_kind: str,
    response_style: str,
    sentences: int,
    max_chars: int = 0,
    extra_instructions: tuple[str, ...] = (),
  ) -> str:
    hint = self._style_instructions.get(response_style, "")
    if sentences > 0:
      brevity = f"，{max_chars}个字以内" if max_chars > 0 else ""
      count_hint = f"[本轮句数] 回复{sentences}句话{brevity}。"
      hint = f"{count_hint}\n{hint}" if hint else f"{count_hint}\n\n"
    route_reference = self._build_route_reference(route_kind)
    if route_reference:
      hint = hint + route_reference
    force_engaging_question = any(
      any(keyword in str(instruction or "") for keyword in ("追问", "继续聊", "互动引导", "顺势往下聊"))
      for instruction in extra_instructions
    )
    if (
      force_engaging_question
      and self._engaging_question_hint
      and route_kind not in ("guard_buy", "gift", "super_chat")
    ):
      hint = self._engaging_question_hint + hint
      return hint
    if (
      response_style not in ("reaction", "guard_thanks", "existential")
      and route_kind not in ("guard_buy", "gift", "super_chat")
      and self._engaging_question_probability > 0
      and random.random() < self._engaging_question_probability
    ):
      hint = self._engaging_question_hint + hint
    return hint

  def _build_route_reference(self, route_kind: str) -> str:
    if route_kind == "guard_buy" and self._guard_thanks_reference:
      return f"[上舰感谢参考]\n{self._guard_thanks_reference}\n\n"
    if route_kind == "gift" and self._gift_thanks_reference:
      return f"[礼物感谢参考]\n{self._gift_thanks_reference}\n\n"
    if route_kind == "super_chat" and self._super_chat_reference:
      return f"[SC 回复参考]\n{self._super_chat_reference}\n\n"
    return ""
