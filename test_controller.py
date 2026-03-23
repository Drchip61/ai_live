"""
LLM Controller 集成测试

覆盖范围：
  1. PromptPlan.from_dict — 正常 JSON / 缺字段 / 非法字段
  2. CommentBrief.to_prompt_line — 各种事件类型的格式化
  3. LLMController._parse_plan — 正常 JSON / markdown 包裹 / 残缺 JSON
  4. LLMController._fallback — 付费事件 / 会员 / 普通弹幕 / 无弹幕
  5. LLMController.dispatch — mock model 正常返回 / 超时 / 异常
  6. _render_prompt — 输入元数据正确填充到模板
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from llm_controller.schema import (
  CommentBrief,
  ControllerInput,
  PromptPlan,
  TopicBrief,
  ViewerBrief,
)
from llm_controller.controller import LLMController


# ================================================================
# PromptPlan.from_dict
# ================================================================

def test_from_dict_normal():
  data = {
    "should_reply": True,
    "urgency": 7,
    "response_style": "detailed",
    "sentences": 3,
    "memory_strategy": "deep_recall",
    "persona_sections": ["童年回忆", "游戏喜好"],
    "corpus_style": "元气",
    "corpus_scene": "互动",
    "knowledge_topics": ["王者荣耀"],
    "topic_assignments": {"c1": "t1", "c2": "new_讨论游戏"},
    "fake_gift_ids": ["c3"],
    "priority": 0,
    "proactive_speak": False,
    "extra_instructions": ["多用比喻"],
  }
  plan = PromptPlan.from_dict(data)
  assert plan.should_reply is True
  assert plan.urgency == 7
  assert plan.response_style == "detailed"
  assert plan.sentences == 3
  assert plan.memory_strategy == "deep_recall"
  assert plan.persona_sections == ("童年回忆", "游戏喜好")
  assert plan.corpus_style == "元气"
  assert plan.knowledge_topics == ("王者荣耀",)
  assert plan.topic_assignments == {"c1": "t1", "c2": "new_讨论游戏"}
  assert plan.fake_gift_ids == ("c3",)
  assert plan.priority == 0
  assert plan.proactive_speak is False
  assert plan.extra_instructions == ("多用比喻",)
  print("  [OK] from_dict 正常解析")


def test_from_dict_defaults():
  plan = PromptPlan.from_dict({})
  assert plan.should_reply is True
  assert plan.urgency == 5
  assert plan.response_style == "normal"
  assert plan.sentences == 2
  assert plan.memory_strategy == "normal"
  assert plan.persona_sections == ()
  assert plan.priority == 1
  print("  [OK] from_dict 缺失字段使用默认值")


def test_from_dict_clamping():
  plan = PromptPlan.from_dict({
    "urgency": 99, "sentences": -5, "priority": 100,
    "response_style": "invalid_style",
    "memory_strategy": "not_a_strategy",
  })
  assert plan.urgency == 9
  assert plan.sentences == 1
  assert plan.priority == 3
  assert plan.response_style == "normal"
  assert plan.memory_strategy == "normal"
  print("  [OK] from_dict 非法值自动钳位/回退默认")


# ================================================================
# CommentBrief.to_prompt_line
# ================================================================

def test_comment_brief_danmaku():
  c = CommentBrief(
    id="1", user_id="u1", nickname="小明",
    content="你好呀", is_new=True, seconds_ago=5.0,
  )
  line = c.to_prompt_line()
  assert "[新]" in line
  assert "小明" in line
  assert "你好呀" in line
  print("  [OK] 普通弹幕格式化")


def test_comment_brief_paid():
  c = CommentBrief(
    id="2", user_id="u2", nickname="大佬",
    content="加油", event_type="super_chat", price=100,
    is_guard_member=True, guard_member_level="舰长",
    is_new=True, seconds_ago=2.0,
  )
  line = c.to_prompt_line()
  assert "super_chat" in line
  assert "¥100" in line
  assert "会员:舰长" in line
  print("  [OK] 付费事件弹幕格式化")


# ================================================================
# LLMController._parse_plan
# ================================================================

def test_parse_plan_json():
  raw = '{"should_reply": true, "urgency": 8, "response_style": "brief", "sentences": 1}'
  plan = LLMController._parse_plan(raw)
  assert plan.should_reply is True
  assert plan.urgency == 8
  assert plan.response_style == "brief"
  print("  [OK] _parse_plan 正常 JSON")


def test_parse_plan_markdown():
  raw = "```json\n{\"should_reply\": false, \"urgency\": 2}\n```"
  plan = LLMController._parse_plan(raw)
  assert plan.should_reply is False
  assert plan.urgency == 2
  print("  [OK] _parse_plan markdown 包裹")


def test_parse_plan_broken_json():
  raw = '{"should_reply": true, "urgency": 6, "sentences": 2'
  plan = LLMController._parse_plan(raw)
  assert plan.should_reply is True
  assert plan.urgency == 6
  print("  [OK] _parse_plan 残缺 JSON (json_repair)")


# ================================================================
# LLMController._fallback
# ================================================================

def test_fallback_paid():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="感谢",
                   event_type="super_chat", price=50, is_new=True),
    ),
    silence_seconds=0,
  )
  plan = LLMController._fallback(inp)
  assert plan.should_reply is True
  assert plan.urgency == 9
  assert plan.priority == 0
  assert plan.route_kind == "super_chat"
  assert plan.response_style == "detailed"
  print("  [OK] fallback 付费事件")


def test_fallback_guard():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="2", user_id="u2", nickname="B", content="问个问题",
                   is_guard_member=True, is_new=True),
    ),
    silence_seconds=0,
  )
  plan = LLMController._fallback(inp)
  assert plan.urgency == 7
  assert plan.memory_strategy == "deep_recall"
  print("  [OK] fallback 会员弹幕")


def test_fallback_normal():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="3", user_id="u3", nickname="C", content="哈哈",
                   is_new=True),
    ),
    silence_seconds=0,
  )
  plan = LLMController._fallback(inp)
  assert plan.should_reply is True
  assert plan.urgency == 5
  assert plan.priority == 1
  print("  [OK] fallback 普通弹幕")


def test_fallback_no_comments_short_silence():
  inp = ControllerInput(comments=(), silence_seconds=5)
  plan = LLMController._fallback(inp)
  assert plan.should_reply is False
  assert plan.proactive_speak is False
  print("  [OK] fallback 无弹幕短沉默")


def test_fallback_no_comments_long_silence():
  inp = ControllerInput(comments=(), silence_seconds=20)
  plan = LLMController._fallback(inp)
  assert plan.proactive_speak is True
  print("  [OK] fallback 无弹幕长沉默→主动发言")


# ================================================================
# LLMController.dispatch (mock model)
# ================================================================

def _make_mock_model(content: str):
  model = MagicMock()
  result = MagicMock()
  result.content = content
  model.ainvoke = AsyncMock(return_value=result)
  return model


def test_dispatch_normal():
  model = _make_mock_model(
    '{"should_reply": true, "urgency": 7, "response_style": "normal", "sentences": 2}'
  )
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="你好",
                   is_new=True),
    ),
    energy=0.8, patience=0.6,
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.should_reply is True
  assert plan.urgency == 7
  assert model.ainvoke.call_count == 1
  print("  [OK] dispatch 正常返回")


def test_dispatch_timeout():
  model = MagicMock()
  async def slow_invoke(*args, **kwargs):
    await asyncio.sleep(10)
  model.ainvoke = slow_invoke
  ctrl = LLMController(model=model, timeout=0.1)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="测试",
                   is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.should_reply is True
  assert plan.urgency == 5
  print("  [OK] dispatch 超时走 fallback")


def test_dispatch_error():
  model = MagicMock()
  model.ainvoke = AsyncMock(side_effect=RuntimeError("模拟错误"))
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="测试",
                   is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.should_reply is True
  print("  [OK] dispatch 异常走 fallback")


# ================================================================
# _render_prompt
# ================================================================

def test_render_prompt():
  model = _make_mock_model("{}")
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    energy=0.82, patience=0.52,
    atmosphere="活跃", emotion="开心",
    stream_phase="直播中", round_count=5,
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="小明",
                   content="你好", is_new=True, seconds_ago=3),
    ),
    comment_rate=2.5, silence_seconds=0,
    viewer_briefs=(
      ViewerBrief(viewer_id="u1", nickname="小明", familiarity=0.5),
    ),
    active_topics=(
      TopicBrief(topic_id="t1", title="游戏讨论", significance=0.7),
    ),
    available_persona_sections=("童年回忆", "游戏喜好"),
    available_knowledge_topics=("王者荣耀",),
    available_corpus_styles=("元气",),
    available_corpus_scenes=("互动",),
    last_response_style="normal",
    last_topic="王者荣耀",
  )
  rendered = ctrl._render_prompt(inp)
  assert "0.82" in rendered
  assert "0.52" in rendered
  assert "小明" in rendered
  assert "游戏讨论" in rendered
  assert "童年回忆" in rendered
  assert "王者荣耀" in rendered
  print("  [OK] _render_prompt 元数据正确填充")


# ================================================================
# Runner
# ================================================================

if __name__ == "__main__":
  tests = [
    ("PromptPlan.from_dict", [
      test_from_dict_normal,
      test_from_dict_defaults,
      test_from_dict_clamping,
    ]),
    ("CommentBrief.to_prompt_line", [
      test_comment_brief_danmaku,
      test_comment_brief_paid,
    ]),
    ("LLMController._parse_plan", [
      test_parse_plan_json,
      test_parse_plan_markdown,
      test_parse_plan_broken_json,
    ]),
    ("LLMController._fallback", [
      test_fallback_paid,
      test_fallback_guard,
      test_fallback_normal,
      test_fallback_no_comments_short_silence,
      test_fallback_no_comments_long_silence,
    ]),
    ("LLMController.dispatch", [
      test_dispatch_normal,
      test_dispatch_timeout,
      test_dispatch_error,
    ]),
    ("_render_prompt", [
      test_render_prompt,
    ]),
  ]

  total, passed, failed = 0, 0, 0
  for group_name, group_tests in tests:
    print(f"\n{'='*60}")
    print(f"  {group_name}")
    print(f"{'='*60}")
    for fn in group_tests:
      total += 1
      try:
        fn()
        passed += 1
      except Exception as e:
        failed += 1
        print(f"  [FAIL] {fn.__name__}: {e}")

  print(f"\n{'='*60}")
  print(f"  结果: {passed}/{total} 通过, {failed} 失败")
  print(f"{'='*60}")
  sys.exit(1 if failed else 0)
