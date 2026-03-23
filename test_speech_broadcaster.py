"""
SpeechBroadcaster 轻量回归测试
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from connection.speech_broadcaster import SpeechBroadcaster


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

def main():
  tests = [
    test_parse_segments_supports_triple_tags,
    test_parse_segments_keeps_legacy_double_tag_compatible,
    test_extract_chinese_strips_triple_tags,
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
