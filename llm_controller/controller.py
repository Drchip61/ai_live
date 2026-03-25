"""
LLM Controller — 集成器架构

规则路由层处理确定性场景（付费事件、入场、沉默），
并行专家组处理 chat 场景：
  ReplyJudge + StyleAdvisor + ContextAdvisor + ActionGuard，
集成器合并所有结果为统一的 PromptPlan。
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import logging
import time
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from .experts import (
  ActionGuard,
  ContextAdvisor,
  ExpertResult,
  ReplyJudge,
  StyleAdvisor,
)
from .rule_router import RuleEnrichment, RuleRouter
from .schema import ControllerInput, PromptPlan

logger = logging.getLogger(__name__)

_CONTROLLER_TIMEOUT = 3.0


def _bump_sentences(value: Any, *, delta: int = 1, minimum: int = 1, maximum: int = 4) -> int:
  try:
    base = int(value)
  except (TypeError, ValueError):
    base = minimum
  return max(minimum, min(maximum, base + delta))


class LLMController:
  """
  集成器：规则优先 → 并行专家 → 合并输出。

  公共接口不变：
    dispatch(ctrl_input) → PromptPlan
    last_plan / last_dispatch_trace / debug_state()
  """

  def __init__(
    self,
    model: Optional[BaseChatModel] = None,
    base_url: str = "http://localhost:2001/v1",
    model_name: str = "qwen3.5-9b",
    timeout: float = _CONTROLLER_TIMEOUT,
    *,
    expert_models: Optional[dict[str, BaseChatModel]] = None,
  ):
    if model is not None:
      resolved_model = model
    elif not str(base_url or "").strip():
      resolved_model = None
    else:
      from langchain_openai import ChatOpenAI
      resolved_model = ChatOpenAI(
        model=model_name,
        api_key="not-needed",
        base_url=base_url,
        temperature=0.3,
        max_tokens=512,
      )

    self._model = resolved_model
    self._timeout = timeout
    self._model_name = model_name

    self._rule_router = RuleRouter()

    expert_map = expert_models or {}
    reply_model = expert_map.get("reply_judge", resolved_model)
    style_model = expert_map.get("style_advisor", resolved_model)
    context_model = expert_map.get("context_advisor", resolved_model)
    action_model = expert_map.get("action_guard", resolved_model)

    self._reply_judge: Optional[ReplyJudge] = (
      ReplyJudge(reply_model, timeout=timeout) if reply_model else None
    )
    self._style_advisor: Optional[StyleAdvisor] = (
      StyleAdvisor(style_model, timeout=timeout) if style_model else None
    )
    self._context_advisor: Optional[ContextAdvisor] = (
      ContextAdvisor(context_model, timeout=timeout) if context_model else None
    )
    self._action_guard: Optional[ActionGuard] = (
      ActionGuard(action_model, timeout=timeout) if action_model else None
    )

    self._last_plan: Optional[PromptPlan] = None
    self._last_dispatch_trace: Optional[dict[str, Any]] = None

  # ----------------------------------------------------------------
  # 公共接口
  # ----------------------------------------------------------------

  @property
  def last_plan(self) -> Optional[PromptPlan]:
    return self._last_plan

  @property
  def last_dispatch_trace(self) -> Optional[dict[str, Any]]:
    return deepcopy(self._last_dispatch_trace) if self._last_dispatch_trace else None

  async def dispatch(
    self,
    ctrl_input: ControllerInput,
    *,
    force_fallback: bool = False,
    fallback_source: str = "fallback_forced",
  ) -> PromptPlan:
    """核心调度：ControllerInput → PromptPlan。"""

    # 1. 规则路由
    rule_plan, enrichment = self._rule_router.route(ctrl_input)

    if force_fallback or self._model is None:
      plan = rule_plan or self._merge_with_rule_defaults(enrichment)
      self._last_plan = plan
      self._remember_dispatch_trace(
        source=fallback_source if force_fallback else "rule_no_model",
        plan=plan,
        enrichment=enrichment,
      )
      return plan

    if rule_plan is not None:
      self._last_plan = rule_plan
      self._remember_dispatch_trace(
        source="rule",
        plan=rule_plan,
        enrichment=enrichment,
      )
      return rule_plan

    # 2. 规则无法决定 → 并行专家组
    started = time.monotonic()
    expert_results = await self._run_experts(ctrl_input, enrichment)
    total_latency = (time.monotonic() - started) * 1000

    # 3. 集成合并
    plan = self._merge(expert_results, enrichment)
    self._last_plan = plan
    self._remember_dispatch_trace(
      source="ensemble",
      plan=plan,
      enrichment=enrichment,
      expert_results=expert_results,
      latency_ms=total_latency,
    )
    return plan

  def debug_state(self) -> dict:
    """调试快照。"""
    plan = self._last_plan
    return {
      "last_plan": plan.to_dict() if plan else None,
      "last_dispatch_trace": self.last_dispatch_trace,
    }

  # ----------------------------------------------------------------
  # 专家并行执行
  # ----------------------------------------------------------------

  async def _run_experts(
    self,
    ctrl_input: ControllerInput,
    enrichment: RuleEnrichment,
  ) -> dict[str, ExpertResult]:
    """并行调用专家组；默认 3 路，必要时可带独立 ActionGuard。"""
    tasks: dict[str, asyncio.Task] = {}

    if self._reply_judge:
      tasks["reply_judge"] = asyncio.create_task(
        self._reply_judge.judge(ctrl_input, enrichment)
      )
    if self._style_advisor:
      tasks["style_advisor"] = asyncio.create_task(
        self._style_advisor.judge(ctrl_input, enrichment)
      )
    if self._context_advisor:
      tasks["context_advisor"] = asyncio.create_task(
        self._context_advisor.judge(ctrl_input, enrichment)
      )
    if self._action_guard:
      tasks["action_guard"] = asyncio.create_task(
        self._action_guard.judge(ctrl_input, enrichment)
      )

    results: dict[str, ExpertResult] = {}
    if tasks:
      done = await asyncio.gather(*tasks.values(), return_exceptions=True)
      for name, result in zip(tasks.keys(), done):
        if isinstance(result, Exception):
          logger.warning("专家 %s 异常: %s", name, result)
          defaults = self._get_expert_defaults(name)
          results[name] = ExpertResult(
            name=name, fields=defaults,
            source="default_exception", error=str(result),
          )
        else:
          results[name] = result

    return results

  @staticmethod
  def _get_expert_defaults(name: str) -> dict[str, Any]:
    if name == "reply_judge":
      return dict(ReplyJudge.DEFAULTS)
    if name == "style_advisor":
      return dict(StyleAdvisor.DEFAULTS)
    if name == "context_advisor":
      return dict(ContextAdvisor.DEFAULTS)
    if name == "action_guard":
      return dict(ActionGuard.DEFAULTS)
    return {}

  # ----------------------------------------------------------------
  # 集成合并
  # ----------------------------------------------------------------

  @staticmethod
  def _merge(
    expert_results: dict[str, ExpertResult],
    enrichment: RuleEnrichment,
  ) -> PromptPlan:
    """将专家组结果 + 规则增强合并为 PromptPlan。"""
    reply = (expert_results.get("reply_judge") or ExpertResult(name="reply_judge")).fields
    style = (expert_results.get("style_advisor") or ExpertResult(name="style_advisor")).fields
    context = (expert_results.get("context_advisor") or ExpertResult(name="context_advisor")).fields
    action_result = expert_results.get("action_guard")
    if action_result is not None:
      action = action_result.fields
    else:
      action = {
        "has_action_request": bool(reply.get("has_action_request", False)),
        "action_hint": str(reply.get("action_hint", "") or ""),
      }
    # 专家组路径只处理“有弹幕但规则层无法直接决定”的场景。
    # 现阶段产品要求：弹幕一律回复，ReplyJudge 只负责给 urgency。
    should_reply = True
    response_style = style.get("response_style", "normal")
    memory_strategy = context.get("memory_strategy", "normal")

    session_anchor = (
      context.get("session_anchor")
      or enrichment.suggested_session_anchor
    )
    instructions = list(context.get("extra_instructions") or ())
    if not instructions:
      instructions = list(enrichment.suggested_extra_instructions)

    action_hint = str(action.get("action_hint", "") or "")
    if action.get("has_action_request") and action_hint:
      instructions.append(
        f"观众要求了你无法执行的操作（{action_hint}），"
        "用角色语气坦率说做不到，不要假装答应"
      )

    corpus_style = str(context.get("corpus_style", "") or "")
    corpus_scene = str(context.get("corpus_scene", "") or "")
    if (
      not should_reply
      or memory_strategy == "deep_recall"
      or enrichment.knowledge_hit
      or enrichment.existential_trigger
      or response_style == "existential"
    ):
      corpus_style = ""
      corpus_scene = ""

    return PromptPlan(
      should_reply=should_reply,
      urgency=reply.get("urgency", 5),
      route_kind="chat",
      response_style=response_style,
      sentences=_bump_sentences(style.get("sentences", 2)),
      tone_hint=style.get("tone_hint", ""),
      memory_strategy=memory_strategy,
      viewer_focus_ids=enrichment.viewer_focus_ids,
      persona_sections=enrichment.persona_sections,
      corpus_style=corpus_style,
      corpus_scene=corpus_scene,
      knowledge_topics=enrichment.knowledge_topics,
      topic_assignments=context.get("topic_assignments", {}),
      fake_gift_ids=enrichment.fake_gift_ids,
      session_mode=context.get("session_mode", "comment_focus"),
      session_anchor=session_anchor,
      priority=1,
      extra_instructions=tuple(instructions),
    )

  @staticmethod
  def _merge_with_rule_defaults(enrichment: RuleEnrichment) -> PromptPlan:
    """无模型时用纯规则信号构建 chat plan。"""
    should_expand = (
      enrichment.has_question
      or enrichment.existential_trigger
      or enrichment.knowledge_hit
      or enrichment.relationship_signal
    )
    return PromptPlan(
      should_reply=True,
      urgency=7 if enrichment.has_guard_member else 5,
      route_kind="chat",
      response_style=(
        "existential" if enrichment.existential_trigger
        else ("detailed" if should_expand else "normal")
      ),
      sentences=3 if should_expand else 2,
      memory_strategy=(
        "deep_recall" if (enrichment.has_guard_member or enrichment.relationship_signal)
        else "normal"
      ),
      viewer_focus_ids=enrichment.viewer_focus_ids,
      persona_sections=enrichment.persona_sections,
      knowledge_topics=enrichment.knowledge_topics,
      fake_gift_ids=enrichment.fake_gift_ids,
      session_mode="comment_focus",
      session_anchor=enrichment.suggested_session_anchor,
      priority=1,
      extra_instructions=enrichment.suggested_extra_instructions,
    )

  # ----------------------------------------------------------------
  # 可观测性
  # ----------------------------------------------------------------

  def _remember_dispatch_trace(
    self,
    *,
    source: str,
    plan: PromptPlan,
    enrichment: RuleEnrichment,
    expert_results: Optional[dict[str, ExpertResult]] = None,
    latency_ms: float = 0.0,
  ) -> None:
    """记录 dispatch 的可观测信息。"""
    experts_trace: dict[str, Any] = {}
    if expert_results:
      for name, result in expert_results.items():
        experts_trace[name] = {
          "source": result.source,
          "latency_ms": round(result.latency_ms, 1),
          "prompt_chars": result.prompt_chars,
          "raw_output": result.raw_output[:300] if result.raw_output else "",
          "error": result.error,
          "fields": result.fields,
        }

    self._last_dispatch_trace = {
      "source": source,
      "model_name": self._model_name,
      "timeout_s": self._timeout,
      "latency_ms": round(max(latency_ms, 0.0), 1),
      "plan_json": plan.to_dict(nested=False),
      "enrichment": {
        "has_guard_member": enrichment.has_guard_member,
        "existential_trigger": enrichment.existential_trigger,
        "knowledge_hit": enrichment.knowledge_hit,
        "competitor_hit": enrichment.competitor_hit,
        "relationship_signal": enrichment.relationship_signal,
        "has_question": enrichment.has_question,
        "persona_sections": list(enrichment.persona_sections),
        "knowledge_topics": list(enrichment.knowledge_topics),
      },
      "experts": experts_trace,
      "error": "",
    }
