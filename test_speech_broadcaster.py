"""
SpeechBroadcaster 轻量回归测试
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from connection.speech_broadcaster import SpeechBroadcaster
from streaming_studio.models import StreamerResponse


def test_parse_segments_supports_triple_tags():
  """三标签回复能拆出 voice_emotion"""
  segments = SpeechBroadcaster._parse_segments("#[Jump][星星][joy]好厉害！")
  assert len(segments) == 1
  seg = segments[0]
  assert seg["motion"] == "Jump"
  assert seg["emotion"] == "星星"
  assert seg["voice_emotion"] == "joy"
  assert seg["text_zh"] == "好厉害！"
  print("  [PASS] 三标签拆段正确")


def test_parse_segments_keeps_legacy_double_tag_compatible():
  """旧双标签仍可解析，并补默认语音情绪"""
  segments = SpeechBroadcaster._parse_segments("#[Wave][happy]欢迎回来")
  assert len(segments) == 1
  seg = segments[0]
  assert seg["motion"] == "Wave"
  assert seg["emotion"] == "happy"
  assert seg["voice_emotion"] == "serenity"
  assert seg["text_zh"] == "欢迎回来"
  print("  [PASS] 旧双标签兼容默认语音情绪")


def test_extract_chinese_strips_triple_tags():
  """三标签 + 双语能正确抽取中文主文本"""
  text = SpeechBroadcaster._extract_chinese(
    "#[Jump][星星][joy]好厉害！ / すごい！#[Idle][- -][serenity]嗯嗯",
  )
  assert text == "好厉害！嗯嗯"
  print("  [PASS] 三标签中文抽取正确")


def test_prepare_segments_keeps_chinese_tts_and_updates_latest_response():
  """prepare 后 text 仍为中文，text_ja 只作为字幕字段暴露"""
  broadcaster = SpeechBroadcaster(
    api_url="http://localhost:9999/say",
    enabled=False,
    translator_enabled=False,
  )
  broadcaster._translator_enabled = True

  async def fake_translate(texts, lang="ja"):
    assert texts == ["好厉害！"]
    assert lang == "ja"
    return ["すごい！"], True

  broadcaster._translate_batch = fake_translate
  response = StreamerResponse(
    content="#[Jump][星星][joy]好厉害！",
    reply_to=(),
    reply_target_text="你之后能叫我霸罢吗",
    nickname="ukpkmkkokk",
  )
  segments = SpeechBroadcaster._parse_segments(response.content)
  SpeechBroadcaster._apply_chinese_speech(segments, response.response_style)
  prepared = asyncio.run(
    broadcaster.prepare_segments_for_broadcast(response, segments)
  )
  seg = prepared[0]
  assert seg["text"] == "好厉害！"
  assert seg["language"] == "Chinese"
  assert seg["text_ja"] == "すごい！"
  assert seg["reply_target_text"] == "你之后能叫我霸罢吗"
  assert seg["nickname"] == "ukpkmkkokk"

  broadcaster._update_latest_response(response)
  latest = broadcaster._latest_response or {}
  assert latest.get("text") == "好厉害！"
  assert latest.get("spoken_text_zh") == "好厉害！"
  assert latest.get("text_ja") == "すごい！"
  assert latest.get("subtitle_text_ja") == "すごい！"
  assert latest.get("subtitle_complete") is True
  assert latest.get("reply_target_text") == "你之后能叫我霸罢吗"
  assert latest.get("nickname") == "ukpkmkkokk"
  print("  [PASS] 中文播报与日语字幕字段分离正确")


def test_prepare_segments_without_translation_keeps_subtitle_empty():
  """翻译失败或关闭时，不再把中文伪装成日语字幕"""
  broadcaster = SpeechBroadcaster(
    api_url="http://localhost:9999/say",
    enabled=False,
    translator_enabled=False,
  )
  response = StreamerResponse(
    content="#[Wave][happy]欢迎回来",
    reply_to=(),
    reply_target_text="欢迎回来",
    nickname="小明",
  )
  segments = SpeechBroadcaster._parse_segments(response.content)
  SpeechBroadcaster._apply_chinese_speech(segments, response.response_style)
  prepared = asyncio.run(
    broadcaster.prepare_segments_for_broadcast(response, segments)
  )
  seg = prepared[0]
  assert seg["text"] == "欢迎回来"
  assert seg["text_ja"] == ""

  broadcaster._update_latest_response(response)
  latest = broadcaster._latest_response or {}
  assert latest.get("text") == "欢迎回来"
  assert latest.get("text_ja") == ""
  assert latest.get("subtitle_complete") is False
  assert latest.get("reply_target_text") == "欢迎回来"
  assert latest.get("nickname") == "小明"
  print("  [PASS] 字幕翻译缺失时保持空字段")

def main():
  tests = [
    test_parse_segments_supports_triple_tags,
    test_parse_segments_keeps_legacy_double_tag_compatible,
    test_extract_chinese_strips_triple_tags,
    test_prepare_segments_keeps_chinese_tts_and_updates_latest_response,
    test_prepare_segments_without_translation_keeps_subtitle_empty,
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
