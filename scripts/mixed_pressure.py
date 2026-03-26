"""
混合压测脚本（本地 stub 版）

覆盖场景：
1. video + 1 条 danmaku + 20 条 entry
2. 3 条 danmaku + 10 条 gift
3. 低优先级长尾生成 + 新弹幕抢占
4. controller expert 超时/丢弃 + fallback

输出：
- controller latency / round total / first sentence enqueue / preempt recovery
- p50 / p95 / p99
- raw / compact 事件规模
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from typing import Iterable
from unittest.mock import AsyncMock, MagicMock

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from llm_controller import LLMController
from llm_controller.schema import CommentBrief, ControllerInput, PromptPlan
from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio
from streaming_studio.config import StudioConfig
from streaming_studio.models import Comment, EventType, StreamerResponse
from streaming_studio.speech_queue import SpeechQueue


@dataclass
class ScenarioSample:
  name: str
  controller_latency_ms: float = 0.0
  round_total_ms: float = 0.0
  first_sentence_enqueue_ms: float = 0.0
  preempt_recovery_ms: float = 0.0
  raw_new_count: int = 0
  compact_new_count: int = 0


class _FakeLLMResult:
  def __init__(self, content: str):
    self.content = content


class _TimedModel:
  def __init__(self, delay_seconds: float, payload: str):
    self._delay_seconds = delay_seconds
    self._payload = payload

  async def ainvoke(self, *_args, **_kwargs):
    await asyncio.sleep(self._delay_seconds)
    return _FakeLLMResult(self._payload)


def _percentile(values: list[float], ratio: float) -> float:
  ordered = sorted(value for value in values if value > 0)
  if not ordered:
    return 0.0
  if len(ordered) == 1:
    return ordered[0]
  index = (len(ordered) - 1) * ratio
  lower = int(index)
  upper = min(lower + 1, len(ordered) - 1)
  weight = index - lower
  return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _metric_summary(samples: Iterable[ScenarioSample], attr: str) -> dict[str, float]:
  values = [float(getattr(sample, attr) or 0.0) for sample in samples]
  return {
    "p50": round(_percentile(values, 0.50), 1),
    "p95": round(_percentile(values, 0.95), 1),
    "p99": round(_percentile(values, 0.99), 1),
  }


def _make_comment(
  *,
  user_id: str,
  nickname: str,
  content: str,
  event_type: EventType = EventType.DANMAKU,
  receive_seq: int,
  received_at: datetime,
  gift_name: str = "",
  gift_num: int = 0,
  price: float = 0.0,
) -> Comment:
  return Comment(
    user_id=user_id,
    nickname=nickname,
    content=content,
    event_type=event_type,
    receive_seq=receive_seq,
    timestamp=received_at,
    received_at=received_at,
    gift_name=gift_name,
    gift_num=gift_num,
    price=price,
  )


def _make_studio() -> StreamingStudio:
  studio = StreamingStudio(
    persona="mio",
    model_type=ModelType.LOCAL_QWEN,
    enable_memory=False,
    enable_global_memory=False,
    enable_topic_manager=False,
    enable_controller=False,
    config=StudioConfig(engaging_question_probability=0.0),
  )
  studio._speech_queue = SpeechQueue(max_size=64)
  studio._speech_broadcaster = MagicMock()
  studio._speech_broadcaster.prepare_segments_for_broadcast = AsyncMock(
    side_effect=lambda response, segments: segments
  )
  studio._speech_broadcaster.cancel_current_playback = MagicMock()
  studio._speech_broadcaster._update_latest_response = MagicMock()
  studio._running = True
  studio.enable_streaming = False
  return studio


async def _run_turn_sample(
  scenario_name: str,
  raw_new_comments: list[Comment],
  plan: PromptPlan,
  *,
  source: str,
  generation_delay_ms: float = 20.0,
) -> ScenarioSample:
  studio = _make_studio()
  studio._runtime_loop = asyncio.get_running_loop()

  async def fake_generate(*_args, **_kwargs):
    await asyncio.sleep(generation_delay_ms / 1000)
    return StreamerResponse(
      content="#[Idle 0][脸黑][neutral]这是压测回复。",
      timestamp=datetime.now(),
    )

  studio._generate_response_with_plan = AsyncMock(side_effect=fake_generate)
  try:
    compact_old, compact_new = studio._build_compact_comment_views([], raw_new_comments)
    started = time.monotonic()
    await studio._generate_and_enqueue_with_plan(
      compact_old,
      compact_new,
      plan,
      source=source,
      raw_old_comments=[],
      raw_new_comments=raw_new_comments,
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    queued = list(studio._speech_queue._items)
    first_enqueue_ms = 0.0
    if queued:
      first_enqueue_ms = max((queued[0].generated_at - started) * 1000, 0.0)

    return ScenarioSample(
      name=scenario_name,
      round_total_ms=elapsed_ms,
      first_sentence_enqueue_ms=first_enqueue_ms,
      raw_new_count=len(raw_new_comments),
      compact_new_count=len(compact_new),
    )
  finally:
    await studio.stop()


async def _scenario_video_danmaku_entry() -> ScenarioSample:
  now = datetime.now()
  comments = [
    _make_comment(
      user_id=f"entry_{idx}",
      nickname=f"进场{idx}",
      content="",
      event_type=EventType.ENTRY,
      receive_seq=idx + 1,
      received_at=now + timedelta(milliseconds=40 * idx),
    )
    for idx in range(20)
  ]
  comments.append(_make_comment(
    user_id="chat_1",
    nickname="弹幕A",
    content="主播先回我这一条",
    event_type=EventType.DANMAKU,
    receive_seq=100,
    received_at=now + timedelta(milliseconds=900),
  ))
  plan = PromptPlan(
    route_kind="chat",
    response_style="normal",
    sentences=1,
    memory_strategy="normal",
    priority=1,
  )
  return await _run_turn_sample(
    "video + 1 danmaku + 20 entry",
    comments,
    plan,
    source="danmaku",
    generation_delay_ms=25.0,
  )


async def _scenario_three_danmaku_ten_gift() -> ScenarioSample:
  now = datetime.now()
  comments: list[Comment] = []
  for idx in range(3):
    comments.append(_make_comment(
      user_id=f"chat_{idx}",
      nickname=f"弹幕{idx}",
      content=f"第{idx + 1}条弹幕",
      event_type=EventType.DANMAKU,
      receive_seq=idx + 1,
      received_at=now + timedelta(milliseconds=60 * idx),
    ))
  for idx in range(10):
    comments.append(_make_comment(
      user_id=f"gift_{idx}",
      nickname=f"老板{idx}",
      content="",
      event_type=EventType.GIFT,
      gift_name="辣条" if idx % 2 == 0 else "荧光棒",
      gift_num=1,
      price=1.0,
      receive_seq=10 + idx,
      received_at=now + timedelta(milliseconds=300 + 50 * idx),
    ))
  plan = PromptPlan(
    route_kind="chat",
    response_style="brief",
    sentences=1,
    memory_strategy="normal",
    priority=1,
  )
  return await _run_turn_sample(
    "3 danmaku + 10 gift",
    comments,
    plan,
    source="danmaku",
    generation_delay_ms=22.0,
  )


async def _scenario_preempt_long_tail_generation() -> ScenarioSample:
  studio = _make_studio()
  studio._runtime_loop = asyncio.get_running_loop()
  low_plan = PromptPlan(
    route_kind="vlm",
    response_style="brief",
    sentences=1,
    memory_strategy="minimal",
    priority=3,
    session_mode="video_focus",
  )
  chat_plan = PromptPlan(
    route_kind="chat",
    response_style="normal",
    sentences=1,
    memory_strategy="normal",
    priority=1,
  )

  async def stubborn_generate(*_args, **_kwargs):
    try:
      await asyncio.sleep(0.18)
    except asyncio.CancelledError:
      pass
    return StreamerResponse(
      content="#[Idle 0][脸黑][neutral]这条低优先级结果应该被丢弃。",
      timestamp=datetime.now(),
    )

  studio._generate_response_with_plan = AsyncMock(side_effect=stubborn_generate)
  try:
    low_task = studio._launch_low_priority_generation([], [], low_plan, source="video", controller_trace=None)

    await asyncio.sleep(0.03)
    preempt_started = time.monotonic()
    danmaku = _make_comment(
      user_id="chat_preempt",
      nickname="抢占弹幕",
      content="新的弹幕来了",
      receive_seq=1,
      received_at=datetime.now(),
    )
    studio.send_comment(danmaku)

    async def fast_generate(*_args, **_kwargs):
      await asyncio.sleep(0.02)
      return StreamerResponse(
        content="#[Idle 0][脸黑][neutral]先处理新的弹幕。",
        timestamp=datetime.now(),
      )

    studio._generate_response_with_plan = AsyncMock(side_effect=fast_generate)
    await studio._generate_and_enqueue_with_plan(
      [],
      [danmaku],
      chat_plan,
      source="danmaku",
      raw_old_comments=[],
      raw_new_comments=[danmaku],
    )
    await asyncio.gather(low_task, return_exceptions=True)

    chat_items = [item for item in studio._speech_queue._items if item.source == "danmaku"]
    first_enqueue_ms = 0.0
    if chat_items:
      first_enqueue_ms = max((chat_items[0].generated_at - preempt_started) * 1000, 0.0)

    return ScenarioSample(
      name="long-tail low-priority + new danmaku preempt",
      round_total_ms=first_enqueue_ms,
      first_sentence_enqueue_ms=first_enqueue_ms,
      preempt_recovery_ms=first_enqueue_ms,
      raw_new_count=1,
      compact_new_count=1,
    )
  finally:
    await studio.stop()


async def _scenario_controller_deadline_drop() -> ScenarioSample:
  slow_reply = _TimedModel(
    0.20,
    '{"urgency": 9, "has_action_request": false, "action_hint": ""}',
  )
  fast_style = _TimedModel(
    0.01,
    '{"response_style": "brief", "sentences": 1}',
  )
  fast_context = _TimedModel(
    0.01,
    '{"memory_strategy": "normal", "session_mode": "comment_focus"}',
  )
  controller = LLMController(
    model=MagicMock(),
    base_url="http://unused.local/v1",
    timeout=0.05,
    expert_models={
      "reply_judge": slow_reply,
      "style_advisor": fast_style,
      "context_advisor": fast_context,
      "action_guard": None,
    },
  )
  ctrl_input = ControllerInput(
    comments=(
      CommentBrief(
        id="c1",
        user_id="u1",
        nickname="观众A",
        content="你怎么看这个视频",
        is_new=True,
      ),
    ),
  )
  started = time.monotonic()
  await controller.dispatch(ctrl_input)
  elapsed_ms = (time.monotonic() - started) * 1000
  trace = controller.last_dispatch_trace or {}
  return ScenarioSample(
    name="controller expert timeout/drop",
    controller_latency_ms=float(trace.get("latency_ms", elapsed_ms) or elapsed_ms),
    round_total_ms=elapsed_ms,
    raw_new_count=1,
    compact_new_count=1,
  )


async def _collect_samples(runs: int) -> list[ScenarioSample]:
  samples: list[ScenarioSample] = []
  for _ in range(runs):
    samples.append(await _scenario_video_danmaku_entry())
    samples.append(await _scenario_three_danmaku_ten_gift())
    samples.append(await _scenario_preempt_long_tail_generation())
    samples.append(await _scenario_controller_deadline_drop())
  return samples


def _print_summary(samples: list[ScenarioSample]) -> None:
  scenario_names = list(dict.fromkeys(sample.name for sample in samples))
  print("\n=== Mixed Pressure Summary ===")
  for scenario_name in scenario_names:
    bucket = [sample for sample in samples if sample.name == scenario_name]
    print(f"\n[{scenario_name}]")
    print(
      "  raw/compact new:",
      f"{round(sum(sample.raw_new_count for sample in bucket) / len(bucket), 1)}",
      "->",
      f"{round(sum(sample.compact_new_count for sample in bucket) / len(bucket), 1)}",
    )
    for metric in (
      "controller_latency_ms",
      "round_total_ms",
      "first_sentence_enqueue_ms",
      "preempt_recovery_ms",
    ):
      summary = _metric_summary(bucket, metric)
      if any(summary.values()):
        print(
          f"  {metric}: "
          f"p50={summary['p50']:.1f}ms "
          f"p95={summary['p95']:.1f}ms "
          f"p99={summary['p99']:.1f}ms"
        )


def main() -> int:
  parser = argparse.ArgumentParser(description="运行本地 stub 混合压测")
  parser.add_argument("--runs", type=int, default=15, help="每个场景重复次数")
  args = parser.parse_args()

  samples = asyncio.run(_collect_samples(max(args.runs, 1)))
  _print_summary(samples)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
