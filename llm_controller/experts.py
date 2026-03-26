"""
并行专家组

四个轻量 LLM 专家，各自负责 PromptPlan 的一小部分字段，
并行执行以降低延迟。每个专家独立超时、独立回退默认值。

- ReplyJudge:    要不要回、多紧急、动作请求检测
- StyleAdvisor:  回复风格、句数、语气
- ContextAdvisor: 记忆策略、会话锚点、动态指令
- ActionGuard:   独立动作检测（已并入 ReplyJudge，仅作备用）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import json_repair
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from prompts import PromptLoader
from .schema import ControllerInput
from .rule_router import RuleEnrichment

logger = logging.getLogger(__name__)

_RENDER_COMMENTS_LIMIT = 8
_RENDER_VIEWERS_LIMIT = 3
_RENDER_TOPICS_LIMIT = 4
_JSON_OBJECT_GUARDRAIL = (
  "\n\n严格输出要求："
  "\n- 只输出一个 JSON 对象本体"
  "\n- 不要输出 markdown 代码块"
  "\n- 不要输出任何解释、前缀、结尾"
  "\n- 不要把 JSON 再包成字符串"
)


# ------------------------------------------------------------------
# 共享工具
# ------------------------------------------------------------------

def _render_lines(lines, *, limit: int, empty_text: str) -> str:
  rendered = [str(l).strip() for l in lines if str(l).strip()]
  if not rendered:
    return empty_text
  visible = rendered[:limit]
  remaining = len(rendered) - len(visible)
  if remaining > 0:
    visible.append(f"... 另有{remaining}项未展开")
  return "\n".join(visible)


def _append_json_object_guardrail(prompt: str) -> str:
  text = str(prompt or "").rstrip()
  if not text:
    return _JSON_OBJECT_GUARDRAIL.strip()
  if _JSON_OBJECT_GUARDRAIL.strip() in text:
    return text
  return f"{text}{_JSON_OBJECT_GUARDRAIL}"


def _normalize_json_text(raw: str) -> str:
  text = str(raw or "").strip()
  text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
  if text.startswith("```"):
    lines = text.split("\n")
    lines = [l for l in lines if not l.strip().startswith("```")]
    text = "\n".join(lines)
  return text.strip()


def _message_content_to_text(content: Any) -> str:
  if isinstance(content, list):
    parts: list[str] = []
    for item in content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict):
        text = item.get("text")
        if text:
          parts.append(str(text))
    return "\n".join(p.strip() for p in parts if str(p).strip()).strip()
  return str(content or "").strip()


def _load_json_candidate(text: str) -> Any:
  try:
    data = json.loads(text)
  except json.JSONDecodeError:
    data = json_repair.loads(text)
  return data


def _json_candidates(text: str) -> list[str]:
  candidates: list[str] = []
  seen: set[str] = set()

  def add(candidate: str) -> None:
    normalized = _normalize_json_text(candidate)
    if normalized and normalized not in seen:
      candidates.append(normalized)
      seen.add(normalized)

  add(text)
  start = text.find("{")
  end = text.rfind("}")
  if start >= 0 and end > start:
    add(text[start:end + 1])
  return candidates


def _parse_json(raw: str) -> dict:
  """从 LLM 原始输出中解析 JSON 对象。"""
  pending = _json_candidates(_normalize_json_text(raw))
  if not pending:
    raise ValueError("专家输出为空")

  seen = set(pending)
  last_error: Optional[Exception] = None
  while pending:
    text = pending.pop(0)
    try:
      data = _load_json_candidate(text)
    except Exception as e:
      last_error = e
      continue

    if isinstance(data, dict):
      return data
    if isinstance(data, str):
      for candidate in _json_candidates(data):
        if candidate not in seen:
          pending.append(candidate)
          seen.add(candidate)
      last_error = ValueError(f"专家输出不是 JSON 对象: {type(data)}")
      continue

    last_error = ValueError(f"专家输出不是 JSON 对象: {type(data)}")

  raise last_error or ValueError("专家输出不是 JSON 对象")


def _format_comments(ctrl_input: ControllerInput) -> tuple[str, int]:
  """格式化弹幕列表，返回 (格式化文本, 新弹幕数)。"""
  comments = ctrl_input.comments
  new_count = sum(1 for c in comments if c.is_new)
  formatted = _render_lines(
    (c.to_prompt_line() for c in comments),
    limit=_RENDER_COMMENTS_LIMIT,
    empty_text="(无弹幕)",
  )
  return formatted, new_count


def _render_catalog(items: tuple[str, ...], *, empty_text: str = "无") -> str:
  rendered = [str(item).strip() for item in items if str(item).strip()]
  return ", ".join(rendered) if rendered else empty_text


def _normalize_catalog_choice(value: Any, available: tuple[str, ...]) -> str:
  text = str(value or "").strip()
  if not text or text.lower() in ("none", "null", "n/a") or text in ("无", "留空", "空", "不触发"):
    return ""
  allowed = {str(item).strip() for item in available if str(item).strip()}
  return text if text in allowed else ""


# ------------------------------------------------------------------
# ExpertResult
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertResult:
  """单个专家的执行结果。"""
  name: str
  fields: dict[str, Any] = field(default_factory=dict)
  source: str = "default"
  latency_ms: float = 0.0
  raw_output: str = ""
  error: str = ""
  prompt_chars: int = 0


# ------------------------------------------------------------------
# ReplyJudge — 回复判官
# ------------------------------------------------------------------

class ReplyJudge:
  """所有弹幕都回，判断多紧急 + 检测动作请求。"""

  DEFAULTS: dict[str, Any] = {
    "should_reply": True,
    "urgency": 5,
    "has_action_request": False,
    "action_hint": "",
  }

  def __init__(self, model: BaseChatModel, timeout: float = 1.5):
    self._model = model
    self._timeout = timeout
    self._prompt_template = PromptLoader().load("controller/reply_judge.txt")

  async def judge(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> ExpertResult:
    prompt = self._render_prompt(ctrl_input, enrichment)
    started = time.monotonic()
    try:
      result = await self._model.ainvoke([HumanMessage(content=prompt)])
      latency_ms = (time.monotonic() - started) * 1000
      raw = _message_content_to_text(getattr(result, "content", ""))
      data = _parse_json(raw)
      fields = {
        "should_reply": True,
        "urgency": max(0, min(9, int(data.get("urgency", 5)))),
        "has_action_request": bool(data.get("has_action_request", False)),
        "action_hint": str(data.get("action_hint", "") or ""),
      }
      return ExpertResult(
        name="reply_judge", fields=fields,
        source="llm", latency_ms=latency_ms,
        raw_output=raw, prompt_chars=len(prompt),
      )
    except Exception as e:
      logger.warning("ReplyJudge 失败: %s", e)
      return ExpertResult(
        name="reply_judge", fields=dict(self.DEFAULTS),
        source="default_error",
        latency_ms=(time.monotonic() - started) * 1000,
        error=str(e), prompt_chars=len(prompt),
      )

  def _render_prompt(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> str:
    formatted_comments, new_count = _format_comments(ctrl_input)
    rate_str = f"{ctrl_input.comment_rate:.1f}" if ctrl_input.comment_rate >= 0 else "未知"
    return self._prompt_template.format(
      formatted_comments=formatted_comments,
      new_count=new_count,
      silence=f"{ctrl_input.silence_seconds:.0f}",
      rate=rate_str,
      has_guard_member="是" if enrichment.has_guard_member else "否",
    )


# ------------------------------------------------------------------
# StyleAdvisor — 风格定调器
# ------------------------------------------------------------------

class StyleAdvisor:
  """决定回复风格、句数、语气色彩。"""

  DEFAULTS: dict[str, Any] = {
    "response_style": "normal",
    "sentences": 2,
    "tone_hint": "",
  }

  def __init__(self, model: BaseChatModel, timeout: float = 1.5):
    self._model = model
    self._timeout = timeout
    self._prompt_template = PromptLoader().load("controller/style_advisor.txt")

  async def judge(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> ExpertResult:
    prompt = self._render_prompt(ctrl_input, enrichment)
    started = time.monotonic()
    try:
      result = await self._model.ainvoke([HumanMessage(content=prompt)])
      latency_ms = (time.monotonic() - started) * 1000
      raw = _message_content_to_text(getattr(result, "content", ""))
      data = _parse_json(raw)
      style = data.get("response_style", "normal")
      valid_styles = {"reaction", "brief", "normal", "detailed", "existential", "guard_thanks"}
      if style not in valid_styles:
        style = "normal"
      fields = {
        "response_style": style,
        "sentences": max(1, min(4, int(data.get("sentences", 2)))),
        "tone_hint": str(data.get("tone_hint", "") or ""),
      }
      return ExpertResult(
        name="style_advisor", fields=fields,
        source="llm", latency_ms=latency_ms,
        raw_output=raw, prompt_chars=len(prompt),
      )
    except Exception as e:
      logger.warning("StyleAdvisor 失败: %s", e)
      return ExpertResult(
        name="style_advisor", fields=dict(self.DEFAULTS),
        source="default_error",
        latency_ms=(time.monotonic() - started) * 1000,
        error=str(e), prompt_chars=len(prompt),
      )

  def _render_prompt(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> str:
    formatted_comments, _ = _format_comments(ctrl_input)

    signals: list[str] = []
    if enrichment.existential_trigger:
      signals.append("存在性问题: 是")
    if enrichment.knowledge_hit:
      topics = ", ".join(enrichment.knowledge_topics) if enrichment.knowledge_topics else "是"
      signals.append(f"知识命中: {topics}")
    if enrichment.has_question:
      signals.append("有提问: 是")
    if enrichment.competitor_hit:
      signals.append("竞品话题: 是")
    signals_text = "\n".join(signals) if signals else "无特殊信号"

    return self._prompt_template.format(
      formatted_comments=formatted_comments,
      energy=f"{ctrl_input.energy:.2f}",
      patience=f"{ctrl_input.patience:.2f}",
      atmosphere=ctrl_input.atmosphere or "正常",
      emotion=ctrl_input.emotion or "正常",
      last_style=ctrl_input.last_response_style,
      signals=signals_text,
    )


# ------------------------------------------------------------------
# ContextAdvisor — 上下文顾问
# ------------------------------------------------------------------

class ContextAdvisor:
  """决定记忆策略、会话锚点、语料触发。"""

  DEFAULTS: dict[str, Any] = {
    "memory_strategy": "normal",
    "session_mode": "comment_focus",
    "session_anchor": "",
    "extra_instructions": [],
    "topic_assignments": {},
    "corpus_style": "",
    "corpus_scene": "",
  }

  def __init__(self, model: BaseChatModel, timeout: float = 1.5):
    self._model = model
    self._timeout = timeout
    self._prompt_template = PromptLoader().load("controller/context_advisor.txt")

  async def judge(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> ExpertResult:
    prompt = _append_json_object_guardrail(
      self._render_prompt(ctrl_input, enrichment)
    )
    started = time.monotonic()
    try:
      result = await self._model.ainvoke([HumanMessage(content=prompt)])
      latency_ms = (time.monotonic() - started) * 1000
      raw = _message_content_to_text(getattr(result, "content", ""))
      data = _parse_json(raw)
      mem = data.get("memory_strategy", "normal")
      if mem not in ("minimal", "normal", "deep_recall"):
        mem = "normal"
      session_mode = data.get("session_mode", "comment_focus")
      if session_mode not in ("none", "comment_focus", "video_focus"):
        session_mode = "comment_focus"
      instructions = data.get("extra_instructions") or []
      if isinstance(instructions, str):
        instructions = [instructions]
      topic_asgn = data.get("topic_assignments") or {}
      if not isinstance(topic_asgn, dict):
        topic_asgn = {}
      corpus_style = _normalize_catalog_choice(
        data.get("corpus_style", ""),
        ctrl_input.available_corpus_styles,
      )
      corpus_scene = _normalize_catalog_choice(
        data.get("corpus_scene", ""),
        ctrl_input.available_corpus_scenes,
      )
      fields = {
        "memory_strategy": mem,
        "session_mode": session_mode,
        "session_anchor": str(data.get("session_anchor", "") or ""),
        "extra_instructions": list(instructions)[:3],
        "corpus_style": corpus_style,
        "corpus_scene": corpus_scene,
        "topic_assignments": {
          str(k): str(v) for k, v in topic_asgn.items()
          if str(k).strip() and str(v).strip()
        },
      }
      return ExpertResult(
        name="context_advisor", fields=fields,
        source="llm", latency_ms=latency_ms,
        raw_output=raw, prompt_chars=len(prompt),
      )
    except Exception as e:
      logger.warning("ContextAdvisor 失败: %s", e)
      return ExpertResult(
        name="context_advisor", fields=dict(self.DEFAULTS),
        source="default_error",
        latency_ms=(time.monotonic() - started) * 1000,
        error=str(e), prompt_chars=len(prompt),
      )

  def _render_prompt(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> str:
    formatted_comments, _ = _format_comments(ctrl_input)
    formatted_viewers = _render_lines(
      (v.to_prompt_line() for v in ctrl_input.viewer_briefs),
      limit=_RENDER_VIEWERS_LIMIT,
      empty_text="(无活跃观众)",
    )
    formatted_topics = _render_lines(
      (t.to_prompt_line() for t in ctrl_input.active_topics),
      limit=_RENDER_TOPICS_LIMIT,
      empty_text="(无活跃话题)",
    )

    signals: list[str] = []
    if enrichment.relationship_signal:
      signals.append("关系牌信号: 是")
    if enrichment.has_guard_member:
      signals.append("有会员: 是")
    if enrichment.knowledge_hit:
      topics = ", ".join(enrichment.knowledge_topics)
      signals.append(f"知识命中: {topics}")
    if enrichment.competitor_hit:
      signals.append("竞品话题: 是")
    signals_text = "\n".join(signals) if signals else "无特殊信号"
    available_corpus_styles = _render_catalog(ctrl_input.available_corpus_styles)
    available_corpus_scenes = _render_catalog(ctrl_input.available_corpus_scenes)

    return self._prompt_template.format(
      formatted_comments=formatted_comments,
      formatted_viewers=formatted_viewers,
      formatted_topics=formatted_topics,
      signals=signals_text,
      available_corpus_styles=available_corpus_styles,
      available_corpus_scenes=available_corpus_scenes,
    )


# ------------------------------------------------------------------
# ActionGuard — 能力边界守卫
# ------------------------------------------------------------------

class ActionGuard:
  """检测弹幕中的动作请求，防止主播承诺做不到的事。"""

  DEFAULTS: dict[str, Any] = {
    "has_action_request": False,
    "action_hint": "",
  }

  def __init__(self, model: BaseChatModel, timeout: float = 1.5):
    self._model = model
    self._timeout = timeout
    self._prompt_template = PromptLoader().load("controller/action_guard.txt")

  async def judge(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> ExpertResult:
    prompt = _append_json_object_guardrail(self._render_prompt(ctrl_input))
    started = time.monotonic()
    try:
      result = await self._model.ainvoke([HumanMessage(content=prompt)])
      latency_ms = (time.monotonic() - started) * 1000
      raw = _message_content_to_text(getattr(result, "content", ""))
      data = _parse_json(raw)
      fields = {
        "has_action_request": bool(data.get("has_action_request", False)),
        "action_hint": str(data.get("action_hint", "") or ""),
      }
      return ExpertResult(
        name="action_guard", fields=fields,
        source="llm", latency_ms=latency_ms,
        raw_output=raw, prompt_chars=len(prompt),
      )
    except Exception as e:
      logger.warning("ActionGuard 失败: %s", e)
      return ExpertResult(
        name="action_guard", fields=dict(self.DEFAULTS),
        source="default_error",
        latency_ms=(time.monotonic() - started) * 1000,
        error=str(e), prompt_chars=len(prompt),
      )

  def _render_prompt(self, ctrl_input: ControllerInput) -> str:
    formatted_comments, _ = _format_comments(ctrl_input)
    return self._prompt_template.format(
      formatted_comments=formatted_comments,
    )
