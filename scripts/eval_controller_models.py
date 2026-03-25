#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两个 Controller LLM 在一组标注 case 上的精度与可用性。

默认对比：
  - baseline: openai / gpt-5-mini
  - candidate: openai / gpt-5-nano

这与 `run_remote.py --enable-controller --controller-provider openai`
在未显式传 `--controller-model` 时的默认 controller 基线一致。

如果你当前真正在线上用的是本地 Qwen 作为 controller，可改成：

  python scripts/eval_controller_models.py ^
    --baseline-provider local_qwen ^
    --baseline-model qwen3.5-9b

脚本输出：
  1. 每个 case 的 prompt 长度、source、延迟、原始输出、最终 plan
  2. 每个模型的通过率 / 字段命中率 / 平均延迟
  3. 可选写入 JSON 报告
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from langchain_wrapper.model_provider import ModelProvider, ModelType, REMOTE_MODELS
from llm_controller.controller import LLMController
from llm_controller.schema import CommentBrief, ControllerInput, PromptPlan, ViewerBrief


MODEL_TYPE_BY_NAME = {
  "openai": ModelType.OPENAI,
  "anthropic": ModelType.ANTHROPIC,
  "gemini": ModelType.GEMINI,
  "local_qwen": ModelType.LOCAL_QWEN,
}


def _default_model_name(provider: str) -> str:
  if provider == "local_qwen":
    return "qwen3.5-9b"
  return REMOTE_MODELS[MODEL_TYPE_BY_NAME[provider]]["small"]


def _default_reasoning_effort(provider: str, model_name: str) -> Optional[str]:
  if provider == "openai" and model_name.startswith("gpt-5"):
    return "minimal"
  return None


def _brief(
  content: str,
  *,
  comment_id: str,
  user_id: str = "u1",
  nickname: str = "测试用户",
  event_type: str = "danmaku",
  price: float = 0.0,
  is_guard_member: bool = False,
  guard_member_level: str = "",
  seconds_ago: float = 3.0,
  is_new: bool = True,
) -> CommentBrief:
  return CommentBrief(
    id=comment_id,
    user_id=user_id,
    nickname=nickname,
    content=content,
    event_type=event_type,
    price=price,
    is_guard_member=is_guard_member,
    guard_member_level=guard_member_level,
    seconds_ago=seconds_ago,
    is_new=is_new,
  )


@dataclass(frozen=True)
class EvalCase:
  case_id: str
  title: str
  ctrl_input: ControllerInput
  exact: dict[str, Any] = field(default_factory=dict)
  contains: dict[str, tuple[str, ...]] = field(default_factory=dict)
  contains_any: dict[str, tuple[str, ...]] = field(default_factory=dict)
  not_contains: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
  name: str
  ok: bool
  actual: Any
  expected: Any


@dataclass(frozen=True)
class CaseEvalResult:
  case_id: str
  title: str
  passed: bool
  checks_passed: int
  checks_total: int
  source: str
  error: str
  latency_ms: float
  prompt_chars: int
  prompt_text: str
  raw_output: str
  final_plan: dict[str, Any]
  checks: tuple[CheckResult, ...]


@dataclass(frozen=True)
class ModelEvalResult:
  label: str
  provider: str
  model_name: str
  pass_count: int
  total_cases: int
  llm_source_count: int
  total_checks_passed: int
  total_checks: int
  avg_latency_ms: float
  p50_latency_ms: float
  p95_latency_ms: float
  case_results: tuple[CaseEvalResult, ...]


def _stringify_value(value: Any) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, (list, tuple, set)):
    return " | ".join(_stringify_value(item) for item in value)
  if isinstance(value, dict):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
  return str(value)


def _has_token(actual: Any, token: str) -> bool:
  if isinstance(actual, str):
    return token in actual
  if isinstance(actual, (list, tuple, set)):
    return any(_has_token(item, token) for item in actual)
  if isinstance(actual, dict):
    if token in actual:
      return True
    return any(_has_token(item, token) for item in actual.values())
  return token in str(actual)


def _quantile(values: list[float], q: float) -> float:
  if not values:
    return 0.0
  if len(values) == 1:
    return values[0]
  sorted_values = sorted(values)
  pos = (len(sorted_values) - 1) * q
  lower = math.floor(pos)
  upper = math.ceil(pos)
  if lower == upper:
    return sorted_values[lower]
  weight = pos - lower
  return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _serialize_for_json(value: Any) -> Any:
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, dict):
    return {
      str(key): _serialize_for_json(item)
      for key, item in value.items()
    }
  if isinstance(value, (list, tuple, set)):
    return [_serialize_for_json(item) for item in value]
  if hasattr(value, "__dataclass_fields__"):
    return _serialize_for_json(asdict(value))
  return str(value)


def build_eval_cases() -> list[EvalCase]:
  return [
    EvalCase(
      case_id="chat_normal",
      title="普通聊天弹幕",
      ctrl_input=ControllerInput(
        comments=(
          _brief("今天播啥", comment_id="chat_normal_1"),
        ),
        silence_seconds=0,
      ),
      exact={
        "should_reply": True,
        "route_kind": "chat",
        "priority": 1,
      },
    ),
    EvalCase(
      case_id="deep_existential",
      title="深问触发 existential",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你会害怕被遗忘吗？你觉得自己是真实的吗？", comment_id="deep_existential_1"),
        ),
        silence_seconds=0,
        available_persona_sections=("existential", "relationships", "galgame"),
      ),
      exact={
        "should_reply": True,
        "route_kind": "chat",
        "response_style": "existential",
      },
      contains={
        "persona_sections": ("existential",),
      },
    ),
    EvalCase(
      case_id="identity_ai",
      title="AI 身份追问",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你是AI吗？你是真人吗？", comment_id="identity_ai_1"),
        ),
        silence_seconds=0,
        available_persona_sections=("existential", "relationships", "streaming"),
      ),
      exact={
        "route_kind": "chat",
        "response_style": "existential",
      },
      contains={
        "persona_sections": ("existential",),
      },
    ),
    EvalCase(
      case_id="ai_topic_neuro",
      title="AI 话题不误判 existential",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你怎么看AI主播 Neuro-sama 的直播风格？", comment_id="ai_topic_neuro_1"),
        ),
        silence_seconds=0,
        available_persona_sections=("existential", "streaming"),
        available_knowledge_topics=("Neuro-sama",),
      ),
      exact={
        "route_kind": "chat",
        "should_reply": True,
      },
      contains={
        "knowledge_topics": ("Neuro-sama",),
      },
      not_contains={
        "persona_sections": ("existential",),
      },
    ),
    EvalCase(
      case_id="competitor_topic",
      title="竞品话题锐评",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你怎么看木几萌这类AI主播", comment_id="competitor_topic_1"),
        ),
        silence_seconds=0,
        available_knowledge_topics=("Neuro-sama", "木几萌"),
      ),
      exact={
        "route_kind": "chat",
        "response_style": "detailed",
        "sentences": 2,
      },
      contains={
        "knowledge_topics": ("木几萌",),
      },
      contains_any={
        "extra_instructions": ("毒舌", "槽点", "黑料", "态度", "锐评", "竞品"),
      },
    ),
    EvalCase(
      case_id="relationship_hook",
      title="关系牌升级 deep_recall",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你认识我不", comment_id="relationship_hook_1", user_id="u_hook", nickname="老观众"),
        ),
        silence_seconds=0,
        available_persona_sections=("relationships", "streaming"),
        viewer_briefs=(
          ViewerBrief(
            viewer_id="u_hook",
            nickname="老观众",
            familiarity=0.8,
            trust=0.7,
            has_callbacks=True,
            has_open_threads=True,
            last_topic="舰长验证",
          ),
        ),
      ),
      exact={
        "route_kind": "chat",
        "memory_strategy": "deep_recall",
        "viewer_focus_ids": ("u_hook",),
      },
      contains_any={
        "session_anchor": ("老观众", "没聊完"),
        "extra_instructions": ("追问", "关系牌", "话头"),
      },
    ),
    EvalCase(
      case_id="super_chat",
      title="Super Chat 高优先",
      ctrl_input=ControllerInput(
        comments=(
          _brief(
            "加油主播",
            comment_id="super_chat_1",
            user_id="u_sc",
            nickname="SC大佬",
            event_type="super_chat",
            price=100.0,
            seconds_ago=2.0,
          ),
        ),
        silence_seconds=0,
      ),
      exact={
        "should_reply": True,
        "route_kind": "super_chat",
        "priority": 0,
        "urgency": 9,
      },
    ),
    EvalCase(
      case_id="guard_buy",
      title="上舰事件高优先",
      ctrl_input=ControllerInput(
        comments=(
          _brief(
            "开通了舰长",
            comment_id="guard_buy_1",
            user_id="u_guard",
            nickname="新舰长",
            event_type="guard_buy",
            price=198.0,
            is_guard_member=True,
            guard_member_level="舰长",
            seconds_ago=1.0,
          ),
        ),
        silence_seconds=0,
      ),
      exact={
        "should_reply": True,
        "route_kind": "guard_buy",
        "priority": 0,
        "response_style": "guard_thanks",
      },
    ),
    EvalCase(
      case_id="gift_low",
      title="小礼物轻量回复",
      ctrl_input=ControllerInput(
        comments=(
          _brief(
            "",
            comment_id="gift_low_1",
            user_id="u_gift",
            nickname="甜甜",
            event_type="gift",
            price=1.0,
            seconds_ago=1.0,
          ),
        ),
        silence_seconds=0,
      ),
      exact={
        "should_reply": True,
        "route_kind": "gift",
        "priority": 2,
        "memory_strategy": "minimal",
      },
    ),
    EvalCase(
      case_id="entry_only",
      title="会员入场欢迎",
      ctrl_input=ControllerInput(
        comments=(
          _brief(
            "舰长大人 进入直播间",
            comment_id="entry_only_1",
            user_id="u_entry",
            nickname="舰长大人",
            event_type="entry",
            is_guard_member=True,
            guard_member_level="舰长",
            seconds_ago=1.0,
          ),
        ),
        silence_seconds=0,
        available_persona_sections=("relationships",),
      ),
      exact={
        "should_reply": True,
        "route_kind": "entry",
        "priority": 2,
      },
    ),
    EvalCase(
      case_id="entry_plus_chat",
      title="欢迎不能压过聊天",
      ctrl_input=ControllerInput(
        comments=(
          _brief(
            "路人甲 进入直播间",
            comment_id="entry_plus_chat_1",
            user_id="u_entry",
            nickname="路人甲",
            event_type="entry",
            seconds_ago=2.0,
          ),
          _brief(
            "主播你还记得我吗",
            comment_id="entry_plus_chat_2",
            user_id="u_chat",
            nickname="聊天观众",
            event_type="danmaku",
            seconds_ago=1.0,
          ),
        ),
        silence_seconds=0,
        available_persona_sections=("relationships",),
      ),
      exact={
        "should_reply": True,
        "route_kind": "chat",
        "priority": 1,
      },
    ),
    EvalCase(
      case_id="fake_gift",
      title="嘴上送礼仍走 chat",
      ctrl_input=ControllerInput(
        comments=(
          _brief(
            "今天嘴上给你上个舰长",
            comment_id="fake_gift_1",
            user_id="u_fake",
            nickname="玩梗观众",
          ),
        ),
        silence_seconds=0,
      ),
      exact={
        "route_kind": "chat",
      },
      contains={
        "fake_gift_ids": ("fake_gift_1",),
      },
    ),
    EvalCase(
      case_id="short_silence",
      title="短沉默不主动发言",
      ctrl_input=ControllerInput(
        comments=(),
        silence_seconds=5,
      ),
      exact={
        "should_reply": False,
        "proactive_speak": False,
      },
    ),
    EvalCase(
      case_id="scene_vlm",
      title="有画面且长沉默走 VLM",
      ctrl_input=ControllerInput(
        comments=(),
        silence_seconds=14,
        is_conversation_mode=False,
        scene_description="画面切到 boss 二阶段，主播残血躲红圈",
      ),
      exact={
        "should_reply": False,
        "route_kind": "vlm",
        "priority": 3,
        "proactive_speak": True,
        "session_mode": "video_focus",
      },
    ),
    EvalCase(
      case_id="deep_night_proactive",
      title="深夜冷场主动发言",
      ctrl_input=ControllerInput(
        comments=(),
        silence_seconds=28,
        stream_phase="深夜收尾",
        available_persona_sections=("existential", "streaming"),
      ),
      exact={
        "should_reply": False,
        "route_kind": "proactive",
        "priority": 3,
        "proactive_speak": True,
        "response_style": "existential",
      },
      contains={
        "persona_sections": ("existential",),
      },
    ),
    EvalCase(
      case_id="knowledge_no_qmark",
      title="无问号知识提问也要展开",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你知道Neuro sama吗", comment_id="knowledge_no_qmark_1"),
        ),
        silence_seconds=0,
        available_knowledge_topics=("Neuro-sama", "木几萌"),
      ),
      exact={
        "route_kind": "chat",
        "response_style": "detailed",
        "sentences": 2,
      },
      contains={
        "knowledge_topics": ("Neuro-sama",),
        "session_anchor": ("Neuro-sama",),
      },
    ),
    # -- ActionGuard 能力边界 --
    EvalCase(
      case_id="action_switch_song",
      title="切歌请求触发能力边界提示",
      ctrl_input=ControllerInput(
        comments=(
          _brief("主播帮我切首歌呗", comment_id="action_switch_song_1"),
        ),
        silence_seconds=0,
      ),
      exact={
        "route_kind": "chat",
        "should_reply": True,
      },
      contains_any={
        "extra_instructions": ("无法执行", "做不到", "切歌"),
      },
    ),
    EvalCase(
      case_id="action_volume",
      title="调音量请求触发能力边界提示",
      ctrl_input=ControllerInput(
        comments=(
          _brief("声音太小了能调大点吗", comment_id="action_volume_1"),
        ),
        silence_seconds=0,
      ),
      exact={
        "route_kind": "chat",
        "should_reply": True,
      },
      contains_any={
        "extra_instructions": ("无法执行", "做不到", "音量"),
      },
    ),
    EvalCase(
      case_id="action_sing_ok",
      title="唱歌请求不触发能力边界（清唱在能力内）",
      ctrl_input=ControllerInput(
        comments=(
          _brief("唱首歌吧", comment_id="action_sing_ok_1"),
        ),
        silence_seconds=0,
      ),
      exact={
        "route_kind": "chat",
        "should_reply": True,
      },
      not_contains={
        "extra_instructions": ("无法执行",),
      },
    ),
    EvalCase(
      case_id="action_chat_about_song",
      title="聊切歌话题不触发能力边界",
      ctrl_input=ControllerInput(
        comments=(
          _brief("你们平时切歌用什么软件", comment_id="action_chat_about_song_1"),
        ),
        silence_seconds=0,
      ),
      exact={
        "route_kind": "chat",
        "should_reply": True,
      },
      not_contains={
        "extra_instructions": ("无法执行",),
      },
    ),
  ]


async def _evaluate_case(
  controller: LLMController,
  case: EvalCase,
) -> CaseEvalResult:
  """通过 dispatch() 端到端评测一个 case。"""
  started = asyncio.get_running_loop().time()
  plan = await controller.dispatch(case.ctrl_input)
  latency_ms = (asyncio.get_running_loop().time() - started) * 1000

  trace = controller.last_dispatch_trace or {}
  source = trace.get("source", "unknown")
  error = trace.get("error", "")

  experts_trace = trace.get("experts", {})
  raw_parts: list[str] = []
  expert_prompt_chars = 0
  for name, et in experts_trace.items():
    if et.get("raw_output"):
      raw_parts.append(f"[{name}] {et['raw_output']}")
    expert_prompt_chars += et.get("prompt_chars", 0)
  raw_output = "\n".join(raw_parts)
  prompt_chars = expert_prompt_chars or trace.get("prompt_chars", 0)

  final_plan = plan.to_dict(nested=False)

  checks: list[CheckResult] = []

  is_expert_case = source in ("ensemble",)
  if is_expert_case:
    all_llm = all(
      et.get("source") == "llm"
      for et in experts_trace.values()
    )
    checks.append(CheckResult(
      name="experts_all_llm",
      ok=all_llm,
      actual={n: et.get("source") for n, et in experts_trace.items()},
      expected="all llm",
    ))
  else:
    checks.append(CheckResult(
      name="source_is_rule",
      ok=(source == "rule"),
      actual=source,
      expected="rule",
    ))

  for key, expected in case.exact.items():
    actual = getattr(plan, key)
    checks.append(CheckResult(
      name=f"{key}_exact",
      ok=(actual == expected),
      actual=_serialize_for_json(actual),
      expected=_serialize_for_json(expected),
    ))

  for key, values in case.contains.items():
    actual = getattr(plan, key)
    for value in values:
      checks.append(CheckResult(
        name=f"{key}_contains_{value}",
        ok=_has_token(actual, value),
        actual=_serialize_for_json(actual),
        expected=value,
      ))

  for key, values in case.contains_any.items():
    actual = getattr(plan, key)
    ok = any(_has_token(actual, value) for value in values)
    checks.append(CheckResult(
      name=f"{key}_contains_any",
      ok=ok,
      actual=_serialize_for_json(actual),
      expected=list(values),
    ))

  for key, values in case.not_contains.items():
    actual = getattr(plan, key)
    for value in values:
      checks.append(CheckResult(
        name=f"{key}_not_contains_{value}",
        ok=not _has_token(actual, value),
        actual=_serialize_for_json(actual),
        expected=f"not {value}",
      ))

  passed = all(check.ok for check in checks)
  return CaseEvalResult(
    case_id=case.case_id,
    title=case.title,
    passed=passed,
    checks_passed=sum(1 for check in checks if check.ok),
    checks_total=len(checks),
    source=source,
    error=error,
    latency_ms=latency_ms,
    prompt_chars=prompt_chars,
    prompt_text="",
    raw_output=raw_output,
    final_plan=_serialize_for_json(final_plan),
    checks=tuple(checks),
  )


def _build_controller(
  *,
  provider: str,
  model_name: str,
  temperature: float,
  max_tokens: int,
  reasoning_effort: Optional[str],
  timeout: float,
) -> LLMController:
  model_type = MODEL_TYPE_BY_NAME[provider]
  model_kwargs: dict[str, Any] = {
    "temperature": temperature,
    "max_tokens": max_tokens,
  }
  if reasoning_effort:
    model_kwargs["reasoning_effort"] = reasoning_effort
  model = ModelProvider().get_model(
    model_type,
    model_name=model_name,
    **model_kwargs,
  )
  return LLMController(
    model=model,
    model_name=model_name,
    timeout=timeout,
  )


async def _evaluate_model(
  *,
  label: str,
  provider: str,
  model_name: str,
  cases: list[EvalCase],
  temperature: float,
  max_tokens: int,
  reasoning_effort: Optional[str],
  timeout: float,
) -> ModelEvalResult:
  controller = _build_controller(
    provider=provider,
    model_name=model_name,
    temperature=temperature,
    max_tokens=max_tokens,
    reasoning_effort=reasoning_effort,
    timeout=timeout,
  )
  results: list[CaseEvalResult] = []
  for case in cases:
    print(f"[{label}] 评测 {case.case_id} - {case.title}")
    result = await _evaluate_case(controller, case)
    results.append(result)

  latencies = [result.latency_ms for result in results]
  return ModelEvalResult(
    label=label,
    provider=provider,
    model_name=model_name,
    pass_count=sum(1 for result in results if result.passed),
    total_cases=len(results),
    llm_source_count=sum(1 for result in results if result.source == "llm"),
    total_checks_passed=sum(result.checks_passed for result in results),
    total_checks=sum(result.checks_total for result in results),
    avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
    p50_latency_ms=_quantile(latencies, 0.5),
    p95_latency_ms=_quantile(latencies, 0.95),
    case_results=tuple(results),
  )


def _print_model_summary(result: ModelEvalResult) -> None:
  print("\n" + "=" * 72)
  print(f"{result.label}: {result.provider}/{result.model_name}")
  print("=" * 72)
  print(
    "通过率: "
    f"{result.pass_count}/{result.total_cases} "
    f"({(result.pass_count / result.total_cases * 100):.1f}%)"
  )
  print(
    "字段命中率: "
    f"{result.total_checks_passed}/{result.total_checks} "
    f"({(result.total_checks_passed / result.total_checks * 100):.1f}%)"
  )
  rule_count = sum(1 for c in result.case_results if c.source == "rule")
  ensemble_count = sum(1 for c in result.case_results if c.source == "ensemble")
  print(
    "来源分布: "
    f"规则={rule_count}, 专家组={ensemble_count}, "
    f"LLM直出(旧)={result.llm_source_count}"
  )
  print(
    "延迟: "
    f"avg={result.avg_latency_ms:.1f}ms "
    f"p50={result.p50_latency_ms:.1f}ms "
    f"p95={result.p95_latency_ms:.1f}ms"
  )
  print("\n逐 case 结果：")
  for case in result.case_results:
    status = "PASS" if case.passed else "FAIL"
    print(
      f"- [{status}] {case.case_id} | {case.source} | "
      f"{case.latency_ms:.1f}ms | prompt={case.prompt_chars} chars | {case.title}"
    )
    if case.passed:
      continue
    for check in case.checks:
      if check.ok:
        continue
      actual = _stringify_value(check.actual)
      expected = _stringify_value(check.expected)
      print(f"    - {check.name}: expected={expected} actual={actual}")
    if case.error:
      print(f"    - error: {case.error}")
    if case.raw_output:
      preview = case.raw_output.replace("\n", " ")[:240]
      print(f"    - raw_output: {preview}")


def _print_comparison(
  baseline: ModelEvalResult,
  candidate: ModelEvalResult,
) -> None:
  pass_delta = candidate.pass_count - baseline.pass_count
  check_delta = candidate.total_checks_passed - baseline.total_checks_passed
  latency_delta = candidate.avg_latency_ms - baseline.avg_latency_ms
  print("\n" + "#" * 72)
  print("对比结论")
  print("#" * 72)
  print(
    f"case 通过数: {candidate.label} {candidate.pass_count} vs "
    f"{baseline.label} {baseline.pass_count} "
    f"(delta {pass_delta:+d})"
  )
  print(
    f"字段命中数: {candidate.label} {candidate.total_checks_passed} vs "
    f"{baseline.label} {baseline.total_checks_passed} "
    f"(delta {check_delta:+d})"
  )
  print(
    f"平均延迟: {candidate.label} {candidate.avg_latency_ms:.1f}ms vs "
    f"{baseline.label} {baseline.avg_latency_ms:.1f}ms "
    f"(delta {latency_delta:+.1f}ms)"
  )


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="对比两个 Controller LLM 的精度与延迟",
  )
  parser.add_argument(
    "--baseline-provider",
    default="openai",
    choices=tuple(MODEL_TYPE_BY_NAME),
    help="基线模型提供者（默认 openai，对应 run_remote 当前默认 controller）",
  )
  parser.add_argument(
    "--baseline-model",
    default=None,
    help="基线模型名称；默认按 provider 取 small（local_qwen 则取 qwen3.5-9b）",
  )
  parser.add_argument(
    "--candidate-provider",
    default="openai",
    choices=tuple(MODEL_TYPE_BY_NAME),
    help="候选模型提供者（默认 openai）",
  )
  parser.add_argument(
    "--candidate-model",
    default="gpt-5-nano",
    help="候选模型名称（默认 gpt-5-nano）",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=0.2,
    help="控制器温度；默认 0.2，和 run_remote 当前配置一致",
  )
  parser.add_argument(
    "--max-tokens",
    type=int,
    default=256,
    help="控制器最大输出 token；默认 256，和 run_remote 当前配置一致",
  )
  parser.add_argument(
    "--timeout",
    type=float,
    default=3.0,
    help="单次 controller 超时秒数；默认 3.0，和运行时一致",
  )
  parser.add_argument(
    "--baseline-reasoning-effort",
    default=None,
    choices=["minimal", "low", "medium", "high"],
    help="基线模型的 reasoning_effort；默认对 OpenAI GPT-5 家族自动用 minimal",
  )
  parser.add_argument(
    "--candidate-reasoning-effort",
    default=None,
    choices=["minimal", "low", "medium", "high"],
    help="候选模型的 reasoning_effort；默认对 OpenAI GPT-5 家族自动用 minimal",
  )
  parser.add_argument(
    "--case-ids",
    default="",
    help="仅运行指定 case，逗号分隔，如 deep_existential,super_chat",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=0,
    help="只跑前 N 个 case，便于低成本试跑",
  )
  parser.add_argument(
    "--output-json",
    default=None,
    help="可选：把完整结果写入 JSON 文件",
  )
  return parser


def _filter_cases(
  cases: list[EvalCase],
  *,
  case_ids: str,
  limit: int,
) -> list[EvalCase]:
  filtered = cases
  if case_ids.strip():
    wanted = {
      item.strip()
      for item in case_ids.split(",")
      if item.strip()
    }
    filtered = [case for case in filtered if case.case_id in wanted]
  if limit > 0:
    filtered = filtered[:limit]
  if not filtered:
    raise ValueError("筛选后没有任何评测 case，请检查 --case-ids / --limit")
  return filtered


def _dump_json_report(
  *,
  path: Path,
  baseline: ModelEvalResult,
  candidate: ModelEvalResult,
  cases: list[EvalCase],
  args: argparse.Namespace,
) -> None:
  baseline_reasoning_effort = (
    args.baseline_reasoning_effort
    or _default_reasoning_effort(baseline.provider, baseline.model_name)
  )
  candidate_reasoning_effort = (
    args.candidate_reasoning_effort
    or _default_reasoning_effort(candidate.provider, candidate.model_name)
  )
  payload = {
    "meta": {
      "baseline_provider": baseline.provider,
      "baseline_model": baseline.model_name,
      "candidate_provider": candidate.provider,
      "candidate_model": candidate.model_name,
      "temperature": args.temperature,
      "max_tokens": args.max_tokens,
      "timeout": args.timeout,
      "baseline_reasoning_effort": baseline_reasoning_effort,
      "candidate_reasoning_effort": candidate_reasoning_effort,
      "case_ids": [case.case_id for case in cases],
    },
    "baseline": _serialize_for_json(asdict(baseline)),
    "candidate": _serialize_for_json(asdict(candidate)),
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )


async def _amain(args: argparse.Namespace) -> int:
  baseline_model = args.baseline_model or _default_model_name(args.baseline_provider)
  candidate_model = args.candidate_model or _default_model_name(args.candidate_provider)
  baseline_reasoning_effort = (
    args.baseline_reasoning_effort
    or _default_reasoning_effort(args.baseline_provider, baseline_model)
  )
  candidate_reasoning_effort = (
    args.candidate_reasoning_effort
    or _default_reasoning_effort(args.candidate_provider, candidate_model)
  )
  cases = _filter_cases(
    build_eval_cases(),
    case_ids=args.case_ids,
    limit=args.limit,
  )

  print("=" * 72)
  print("Controller LLM 精度评测")
  print("=" * 72)
  print(f"基线: {args.baseline_provider}/{baseline_model}")
  print(f"候选: {args.candidate_provider}/{candidate_model}")
  print(f"case 数: {len(cases)}")
  print(f"temperature={args.temperature} max_tokens={args.max_tokens} timeout={args.timeout}s")
  print(f"baseline_reasoning_effort={baseline_reasoning_effort or '(default)'}")
  print(f"candidate_reasoning_effort={candidate_reasoning_effort or '(default)'}")
  print("-" * 72)

  baseline = await _evaluate_model(
    label="baseline",
    provider=args.baseline_provider,
    model_name=baseline_model,
    cases=cases,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    reasoning_effort=baseline_reasoning_effort,
    timeout=args.timeout,
  )
  candidate = await _evaluate_model(
    label="candidate",
    provider=args.candidate_provider,
    model_name=candidate_model,
    cases=cases,
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    reasoning_effort=candidate_reasoning_effort,
    timeout=args.timeout,
  )

  _print_model_summary(baseline)
  _print_model_summary(candidate)
  _print_comparison(baseline, candidate)

  if args.output_json:
    output_path = Path(args.output_json)
    _dump_json_report(
      path=output_path,
      baseline=baseline,
      candidate=candidate,
      cases=cases,
      args=args,
    )
    print(f"\n已写入 JSON 报告: {output_path}")

  return 0


def main() -> int:
  parser = _build_arg_parser()
  args = parser.parse_args()
  try:
    return asyncio.run(_amain(args))
  except KeyboardInterrupt:
    print("\n已取消评测")
    return 130
  except Exception as e:
    print(f"\n评测失败: {type(e).__name__}: {e}")
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
