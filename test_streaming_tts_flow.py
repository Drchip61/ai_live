"""
流式 TTS / 聊天抢占 轻量回归测试
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from streaming_studio.config import StudioConfig
from streaming_studio.models import Comment, EventType, ResponseChunk, StreamerResponse
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


def main():
  tests = [
    test_sentence_streamer_backfills_reply_target_and_segment_total,
    test_entry_reply_target_text_is_blank,
    test_wait_for_response_continuation_blocks_entry_gap,
    test_send_comment_preempts_low_priority_playback,
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
