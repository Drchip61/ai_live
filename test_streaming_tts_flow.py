"""
流式 TTS / 聊天抢占 轻量回归测试
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from connection.speech_broadcaster import SpeechBroadcaster
from langchain_wrapper import ModelType
from streaming_studio.config import StudioConfig
from streaming_studio.models import Comment, EventType, ResponseChunk, StreamerResponse
from streaming_studio.speech_queue import SpeechQueue
from streaming_studio.speech_queue import SpeechItem, SpeechQueue
from streaming_studio.studio import StreamingStudio, _SentenceStreamer


def test_sentence_streamer_backfills_reply_target_and_segment_total():
  """流式句级入队应保留回复目标元数据，并在收尾后补齐 segment_total。"""

  async def scenario():
    queue = SpeechQueue(max_size=10)
    streamer = _SentenceStreamer(
      speech_queue=queue,
      speech_broadcaster=MagicMock(),
      response_id="resp_stream",
      priority=1,
      ttl=30,
      source="danmaku",
      comments=[],
      reply_target_text="你之后能叫我霸罢吗",
      reply_target_nickname="ukpkmkkokk",
    )
    response = StreamerResponse(
      id="resp_stream",
      content=(
        "#[Idle 0][脸黑][neutral]第一句先回你。"
        "#[Happy 0][星星][joy]第二句继续接话。"
      ),
      reply_target_text="你之后能叫我霸罢吗",
      nickname="ukpkmkkokk",
    )

    streamer.on_chunk(ResponseChunk(
      response_id=response.id,
      chunk=response.content,
      accumulated=response.content,
      done=False,
    ))
    await streamer.flush(response)

    assert len(streamer.pushed_items) == 2
    assert all(not item.is_last_segment for item in streamer.pushed_items)
    assert all(item.response_id == "resp_stream" for item in streamer.pushed_items)
    streamer.bind_response(response)

    first, second = streamer.pushed_items
    assert first.segment["reply_target_text"] == "你之后能叫我霸罢吗"
    assert second.segment["reply_target_text"] == "你之后能叫我霸罢吗"
    assert first.segment["nickname"] == "ukpkmkkokk"
    assert second.segment["nickname"] == "ukpkmkkokk"
    assert first.segment_total == 2
    assert second.segment_total == 2
    assert first.response_id == "resp_stream"
    assert second.response_id == "resp_stream"
    assert first.is_last_segment is False
    assert second.is_last_segment is True

  asyncio.run(scenario())
  print("  [PASS] 流式句级入队会补齐 reply_target 与 segment_total")


def test_entry_reply_target_text_is_blank():
  """入场欢迎不应污染 reply_target_text。"""
  entry = Comment(
    user_id="u_entry",
    nickname="新观众",
    content="欢迎进直播间",
    event_type=EventType.ENTRY,
  )
  assert StreamingStudio._reply_target_text(entry) == ""
  print("  [PASS] entry 不再写入 reply_target_text")


def test_wait_for_response_continuation_blocks_entry_gap():
  """同一条多句回复未接上时，Dispatcher 应短暂等后句，别让 entry 插中间。"""

  async def scenario():
    studio = StreamingStudio(
      persona="mio",
      model_type=ModelType.LOCAL_QWEN,
      enable_memory=False,
      enable_global_memory=False,
      enable_topic_manager=False,
      enable_controller=False,
      config=StudioConfig(engaging_question_probability=0.0),
    )
    studio._speech_queue = SpeechQueue(max_size=10)
    studio._running = True
    response = StreamerResponse(id="resp_keep", content="第一句。第二句。")
    current_item = SpeechItem(
      segment={"text_zh": "第一句。"},
      priority=1,
      ttl=30,
      source="danmaku",
      response_id="resp_keep",
      response=response,
      segment_index=0,
      segment_total=2,
    )
    waiter = asyncio.create_task(
      studio._wait_for_response_continuation(current_item, timeout_seconds=0.3)
    )
    await asyncio.sleep(0.05)
    await studio._speech_queue.push(SpeechItem(
      segment={"text_zh": "欢迎回来"},
      priority=2,
      ttl=30,
      source="entry",
      response_id="entry_resp",
      response=StreamerResponse(content="欢迎回来"),
    ))
    assert waiter.done() is False
    await asyncio.sleep(0.05)
    await studio._speech_queue.push(SpeechItem(
      segment={"text_zh": "第二句。"},
      priority=1,
      ttl=30,
      source="danmaku",
      response_id="resp_keep",
      response=response,
      segment_index=1,
      segment_total=2,
    ))
    await waiter

  asyncio.run(scenario())
  print("  [PASS] Dispatcher 会优先等同一 response 的后续句子")


def test_send_comment_preempts_low_priority_playback():
  """新弹幕到来时应立即解除低优先级视频/独白播报阻塞。"""

  async def scenario():
    studio = StreamingStudio(
      persona="mio",
      model_type=ModelType.LOCAL_QWEN,
      enable_memory=False,
      enable_global_memory=False,
      enable_topic_manager=False,
      enable_controller=False,
      config=StudioConfig(engaging_question_probability=0.0),
    )
    studio.database = MagicMock()
    studio._speech_broadcaster = MagicMock()
    studio._current_dispatch_source = "video"

    studio.send_comment(Comment(
      user_id="u1",
      nickname="观众A",
      content="主播先回我这条",
    ))

    studio._speech_broadcaster.cancel_current_playback.assert_called_once()

  asyncio.run(scenario())
  print("  [PASS] 新弹幕会抢占低优先级视频播报")


def test_stale_low_priority_generation_is_dropped_even_if_cancel_is_swallowed():
  """低优先级生成若吞掉 CancelledError，stale guard 仍应丢弃结果。"""

  async def scenario():
    studio = StreamingStudio(
      persona="mio",
      model_type=ModelType.LOCAL_QWEN,
      enable_memory=False,
      enable_global_memory=False,
      enable_topic_manager=False,
      enable_controller=False,
      config=StudioConfig(engaging_question_probability=0.0),
    )
    studio._running = True
    studio._runtime_loop = asyncio.get_running_loop()
    studio._speech_queue = SpeechQueue(max_size=10)
    studio._speech_broadcaster = MagicMock()
    studio._speech_broadcaster.prepare_segments_for_broadcast = AsyncMock(
      side_effect=lambda response, segments: segments
    )
    studio._speech_broadcaster.cancel_current_playback = MagicMock()
    studio._speech_broadcaster._update_latest_response = MagicMock()

    async def stubborn_generation(*args, **kwargs):
      try:
        await asyncio.sleep(0.05)
      except asyncio.CancelledError:
        pass
      return StreamerResponse(
        content="#[Idle 0][脸黑][neutral]这句低优先级回复不该入队。",
        timestamp=datetime.now(),
      )

    studio._generate_response_with_plan = AsyncMock(side_effect=stubborn_generation)
    plan = type("Plan", (), {
      "priority": 3,
      "route_kind": "vlm",
      "response_style": "brief",
      "sentences": 1,
      "memory_strategy": "minimal",
    })()

    task = studio._launch_low_priority_generation([], [], plan, source="video", controller_trace=None)
    await asyncio.sleep(0.01)
    studio.send_comment(Comment(
      user_id="u1",
      nickname="观众A",
      content="先回我这条弹幕",
      event_type=EventType.DANMAKU,
    ))
    await asyncio.sleep(0.1)
    await asyncio.gather(task, return_exceptions=True)

    assert studio._speech_queue.size == 0
    assert studio._response_queue.empty()
    assert studio._generation_preempt_count >= 1

  asyncio.run(scenario())
  print("  [PASS] 低优先级生成即使吞掉 cancel，也会被 stale guard 丢弃")


def test_wait_for_playback_returns_interrupted():
  """播放被抢占时，wait_for_playback 应返回 interrupted。"""

  async def scenario():
    broadcaster = SpeechBroadcaster("http://127.0.0.1:9999", enabled=False)
    broadcaster._playback_done.clear()
    broadcaster.cancel_current_playback()
    state = await broadcaster.wait_for_playback()
    assert state == "interrupted"

  asyncio.run(scenario())
  print("  [PASS] wait_for_playback 可区分 interrupted")


def test_popped_low_priority_item_is_dropped_after_danmaku_preempt():
  """低优先级条目即使已 pop，只要抢占 epoch 变了也不应再发送。"""

  async def scenario():
    studio = StreamingStudio(
      persona="mio",
      model_type=ModelType.LOCAL_QWEN,
      enable_memory=False,
      enable_global_memory=False,
      enable_topic_manager=False,
      enable_controller=False,
      config=StudioConfig(engaging_question_probability=0.0),
    )
    studio._running = True
    studio._runtime_loop = asyncio.get_running_loop()
    studio.database = MagicMock()
    studio._speech_queue = SpeechQueue(max_size=10)
    studio._speech_broadcaster = MagicMock()
    studio._speech_broadcaster.send_segment = AsyncMock(return_value=True)
    studio._speech_broadcaster.wait_for_playback = AsyncMock(return_value="completed")
    studio._speech_broadcaster.cancel_current_playback = MagicMock()
    studio._speech_broadcaster._update_latest_response = MagicMock()

    low_response = StreamerResponse(id="resp_low", content="这句欢迎词不该漏播")
    await studio._speech_queue.push(SpeechItem(
      segment={"text_zh": "这句欢迎词不该漏播"},
      priority=3,
      ttl=30,
      source="video",
      response_id=low_response.id,
      response=low_response,
      preempt_epoch=studio._low_priority_preempt_epoch,
    ))

    original_pop = studio._speech_queue.pop

    async def pop_then_preempt():
      item = await original_pop()
      if item is not None:
        studio.send_comment(Comment(
          user_id="u1",
          nickname="观众A",
          content="先回我这条弹幕",
          event_type=EventType.DANMAKU,
        ))
        studio._running = False
      return item

    studio._speech_queue.pop = pop_then_preempt
    await studio._tts_dispatch_loop()
    await asyncio.sleep(0.05)

    assert studio._speech_broadcaster.send_segment.await_count == 0
    assert studio._low_priority_preempt_epoch >= 1

  asyncio.run(scenario())
  print("  [PASS] 已 pop 的低优先级条目也会因抢占 epoch 被丢弃")


def test_pipeline_round_logs_prompt_timing_fields():
  """pipeline_round 应带上 prompt/retrieve/compose 等细粒度 timing。"""

  studio = StreamingStudio(
    persona="mio",
    model_type=ModelType.LOCAL_QWEN,
    enable_memory=False,
    enable_global_memory=False,
    enable_topic_manager=False,
    enable_controller=False,
    config=StudioConfig(engaging_question_probability=0.0),
  )
  fake_memory_mgr = type("FakeMemoryManager", (), {
    "_read_backlog": 2,
    "_refresh_backlog": 1,
    "_refresh_merged": 0,
    "_store_backlog": 0,
  })()
  studio.llm_wrapper = MagicMock(memory_manager=fake_memory_mgr)
  captured: list[str] = []
  studio._timing_log = MagicMock()
  studio._timing_log.info.side_effect = lambda payload: captured.append(payload)

  studio._timer.start_round()
  studio._timer.mark("LLM生成")
  timings = studio._timer.finish()
  comment = Comment(user_id="u1", nickname="观众A", content="你还记得上次那首歌吗")
  plan = type("Plan", (), {
    "route_kind": "chat",
    "response_style": "normal",
    "sentences": 2,
    "memory_strategy": "normal",
    "session_mode": "comment_focus",
    "session_anchor": "继续上次的话题",
    "priority": 1,
  })()
  response = StreamerResponse(
    id="resp_timing",
    content="记得呀，我们上次聊到副歌那段。",
    response_style="normal",
    reply_target_text="你还记得上次那首歌吗",
    nickname="观众A",
    timing_trace={
      "effective_memory_strategy": "deep_recall",
      "prompt_prep_ms": 12.3,
      "retrieve_ms": 456.7,
      "compose_ms": 34.5,
      "resolve_prompt_ms": 503.5,
      "llm_first_token_ms": 210.4,
      "response_total_ms": 1888.0,
    },
  )

  studio._log_pipeline_timing(
    timings,
    source="danmaku",
    old_comments=[],
    new_comments=[comment],
    plan=plan,
    response=response,
  )

  assert captured, "timing logger 应写出至少一条 pipeline_round"
  payload = json.loads(captured[-1])
  assert payload["plan"]["effective_memory_strategy"] == "deep_recall"
  assert payload["backlog"]["memory_read_queue"] == 2
  assert payload["response"]["timing"]["prompt_prep_ms"] == 12.3
  assert payload["response"]["timing"]["retrieve_ms"] == 456.7
  assert payload["response"]["timing"]["compose_ms"] == 34.5
  assert payload["response"]["timing"]["resolve_prompt_ms"] == 503.5
  assert payload["response"]["timing"]["llm_first_token_ms"] == 210.4
  assert payload["response"]["timing"]["response_total_ms"] == 1888.0
  print("  [PASS] pipeline_round 已带上 prompt/retrieve/compose 细粒度 timing")


def test_compact_entry_and_gift_events_for_controller_view():
  """entry/gift 在时间窗内应压成 compact 视图，但保留原始事件供副作用使用。"""

  studio = StreamingStudio(
    persona="mio",
    model_type=ModelType.LOCAL_QWEN,
    enable_memory=False,
    enable_global_memory=False,
    enable_topic_manager=False,
    enable_controller=False,
    config=StudioConfig(engaging_question_probability=0.0),
  )
  now = datetime.now()
  entry_a = Comment(
    user_id="u_entry_a",
    nickname="观众A",
    content="",
    event_type=EventType.ENTRY,
    received_at=now,
    timestamp=now,
    receive_seq=1,
  )
  entry_b = Comment(
    user_id="u_entry_b",
    nickname="观众B",
    content="",
    event_type=EventType.ENTRY,
    received_at=now + timedelta(milliseconds=400),
    timestamp=now + timedelta(milliseconds=400),
    receive_seq=2,
  )
  gift_a = Comment(
    user_id="u_gift_a",
    nickname="老板A",
    content="",
    event_type=EventType.GIFT,
    gift_name="辣条",
    gift_num=2,
    received_at=now + timedelta(seconds=1),
    timestamp=now + timedelta(seconds=1),
    receive_seq=3,
  )
  gift_b = Comment(
    user_id="u_gift_b",
    nickname="老板B",
    content="",
    event_type=EventType.GIFT,
    gift_name="荧光棒",
    gift_num=3,
    received_at=now + timedelta(seconds=1, milliseconds=300),
    timestamp=now + timedelta(seconds=1, milliseconds=300),
    receive_seq=4,
  )

  compact_old, compact_new = studio._build_compact_comment_views([], [entry_a, entry_b, gift_a, gift_b])

  assert compact_old == []
  assert len(compact_new) == 2
  assert compact_new[0].event_type == EventType.ENTRY
  assert compact_new[1].event_type == EventType.GIFT
  assert "等2位观众" in compact_new[0].content
  assert compact_new[1].gift_num == 5
  assert compact_new[1].gift_name == "礼物合集"
  print("  [PASS] entry/gift 会压成 compact controller 视图")


def main():
  tests = [
    test_sentence_streamer_backfills_reply_target_and_segment_total,
    test_entry_reply_target_text_is_blank,
    test_wait_for_response_continuation_blocks_entry_gap,
    test_send_comment_preempts_low_priority_playback,
    test_stale_low_priority_generation_is_dropped_even_if_cancel_is_swallowed,
    test_wait_for_playback_returns_interrupted,
    test_popped_low_priority_item_is_dropped_after_danmaku_preempt,
    test_pipeline_round_logs_prompt_timing_fields,
    test_compact_entry_and_gift_events_for_controller_view,
  ]

  failed = 0
  for test_fn in tests:
    try:
      test_fn()
    except AssertionError as e:
      failed += 1
      print(f"  [FAIL] {test_fn.__name__}: {e}")
    except Exception as e:
      failed += 1
      print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")

  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
