"""
LLM Controller 集成测试（集成器架构版）

覆盖范围：
  1. PromptPlan.from_dict — 正常 JSON / 缺字段 / 非法字段
  2. CommentBrief.to_prompt_line — 各种事件类型的格式化
  3. RuleRouter.route — 付费事件 / 会员 / 入场 / 弹幕 / 沉默
  4. RuleRouter 规则增强 — persona_sections / knowledge / fake_gift 信号
  5. LLMController.dispatch — 规则路由 / 专家并行 / 集成合并 / 超时 / 异常
  6. _merge — 专家结果合并 + enrichment 回退
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
from llm_controller.rule_router import RuleRouter
from llm_controller.experts import ContextAdvisor, ExpertResult, ReplyJudge


# ================================================================
# PromptPlan.from_dict
# ================================================================

def test_from_dict_normal():
  data = {
    "should_reply": True,
    "urgency": 7,
    "route_kind": "super_chat",
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
  assert plan.priority == 1
  assert plan.response_style == "normal"
  assert plan.memory_strategy == "normal"
  print("  [OK] from_dict 非法值自动钳位/回退默认")


def test_from_dict_route_priority_normalization():
  entry_plan = PromptPlan.from_dict({"route_kind": "entry", "priority": 0})
  chat_plan = PromptPlan.from_dict({"route_kind": "chat", "priority": 3})
  vlm_plan = PromptPlan.from_dict({"route_kind": "vlm", "priority": 1})
  assert entry_plan.priority == 2
  assert chat_plan.priority == 1
  assert vlm_plan.priority == 3
  print("  [OK] from_dict 按路由归一化优先级")


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
# RuleRouter.route — 确定性场景
# ================================================================

_router = RuleRouter()


def test_rule_paid():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="感谢",
                   event_type="super_chat", price=50, is_new=True),
    ),
    silence_seconds=0,
  )
  plan, enrichment = _router.route(inp)
  assert plan is not None
  assert plan.should_reply is True
  assert plan.urgency == 9
  assert plan.priority == 0
  assert plan.route_kind == "super_chat"
  assert plan.response_style == "detailed"
  print("  [OK] 规则路由: 付费事件")


def test_rule_guard():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="2", user_id="u2", nickname="B", content="问个问题",
                   event_type="danmaku", is_guard_member=True, is_new=True),
    ),
    silence_seconds=0,
  )
  plan, enrichment = _router.route(inp)
  assert plan is None, "有弹幕应交给专家组"
  assert enrichment.has_guard_member is True
  print("  [OK] 规则路由: 会员弹幕→专家组 (enrichment 正确)")


def test_rule_normal_danmaku():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="3", user_id="u3", nickname="C", content="哈哈",
                   is_new=True),
    ),
    silence_seconds=0,
  )
  plan, enrichment = _router.route(inp)
  assert plan is None, "普通弹幕应交给专家组"
  print("  [OK] 规则路由: 普通弹幕→专家组")


def test_rule_guard_entry():
  inp = ControllerInput(
    comments=(
      CommentBrief(
        id="entry_guard_1", user_id="u4", nickname="舰长大人",
        content="舰长大人 进入直播间", event_type="entry",
        is_guard_member=True, guard_member_level="舰长", is_new=True,
      ),
    ),
    silence_seconds=0,
    available_persona_sections=("relationships",),
  )
  plan, enrichment = _router.route(inp)
  assert plan is not None
  assert plan.route_kind == "entry"
  assert plan.urgency == 6
  assert plan.response_style == "normal"
  assert plan.sentences == 3
  assert plan.memory_strategy == "normal"
  assert plan.priority == 2
  assert plan.persona_sections == ("relationships",)
  assert plan.extra_instructions == ("这是会员进场欢迎，要比普通入场更热情，点名并点出等级，但不要误说成新上舰。",)
  print("  [OK] 规则路由: 会员入场")


def test_rule_entry_with_danmaku_goes_to_experts():
  inp = ControllerInput(
    comments=(
      CommentBrief(
        id="entry_1", user_id="u_entry", nickname="路人甲",
        content="路人甲 进入直播间", event_type="entry", is_new=True,
      ),
      CommentBrief(
        id="chat_1", user_id="u_chat", nickname="聊天观众",
        content="主播你还记得我吗", event_type="danmaku", is_new=True,
      ),
    ),
    silence_seconds=0,
    available_persona_sections=("relationships",),
  )
  plan, enrichment = _router.route(inp)
  assert plan is None, "有弹幕时 entry 不应抢路由"
  assert enrichment.relationship_signal is True
  print("  [OK] 规则路由: entry+弹幕→专家组")


def test_rule_no_comments_short_silence():
  inp = ControllerInput(comments=(), silence_seconds=5)
  plan, enrichment = _router.route(inp)
  assert plan is not None
  assert plan.should_reply is False
  assert plan.proactive_speak is False
  print("  [OK] 规则路由: 无弹幕短沉默→跳过")


def test_rule_no_comments_long_silence():
  inp = ControllerInput(comments=(), silence_seconds=20)
  plan, enrichment = _router.route(inp)
  assert plan is not None
  assert plan.proactive_speak is True
  print("  [OK] 规则路由: 无弹幕长沉默→主动发言")


def test_rule_vlm_scene():
  inp = ControllerInput(
    comments=(), silence_seconds=14,
    is_conversation_mode=False,
    scene_description="画面切到 boss 二阶段",
  )
  plan, enrichment = _router.route(inp)
  assert plan is not None
  assert plan.route_kind == "vlm"
  assert plan.proactive_speak is True
  assert plan.session_mode == "video_focus"
  print("  [OK] 规则路由: 有画面+长沉默→VLM")


# ================================================================
# RuleRouter 规则增强信号
# ================================================================

def test_enrichment_existential():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你是AI吗？", is_new=True),
    ),
    available_persona_sections=("existential", "streaming"),
  )
  _, enrichment = _router.route(inp)
  assert enrichment.existential_trigger is True
  assert "existential" in enrichment.persona_sections
  print("  [OK] 增强信号: existential 触发")


def test_enrichment_knowledge():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你知道Neuro-sama吗", is_new=True),
    ),
    available_knowledge_topics=("Neuro-sama", "木几萌"),
  )
  _, enrichment = _router.route(inp)
  assert enrichment.knowledge_hit is True
  assert "Neuro-sama" in enrichment.knowledge_topics
  print("  [OK] 增强信号: knowledge 命中")


def test_enrichment_fake_gift():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="fg1", user_id="u1", nickname="A",
                   content="嘴上给你上个舰长", is_new=True),
    ),
  )
  _, enrichment = _router.route(inp)
  assert "fg1" in enrichment.fake_gift_ids
  print("  [OK] 增强信号: 假礼物检测")


def test_enrichment_relationship():
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u_hook", nickname="老观众",
                   content="你还记得我吗", is_new=True),
    ),
    viewer_briefs=(
      ViewerBrief(
        viewer_id="u_hook", nickname="老观众",
        familiarity=0.8, has_open_threads=True, last_topic="舰长验证",
      ),
    ),
    available_persona_sections=("relationships",),
  )
  _, enrichment = _router.route(inp)
  assert enrichment.relationship_signal is True
  assert "u_hook" in enrichment.viewer_focus_ids
  assert enrichment.suggested_session_anchor != ""
  print("  [OK] 增强信号: 关系牌检测")


# ================================================================
# LLMController.dispatch — 规则路由路径
# ================================================================

def test_dispatch_rule_route():
  """付费事件走规则，不调用专家。"""
  model = MagicMock()
  model.ainvoke = AsyncMock(side_effect=RuntimeError("不应被调用"))
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="感谢",
                   event_type="guard_buy", is_guard_member=True, is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.route_kind == "guard_buy"
  assert plan.priority == 0
  model.ainvoke.assert_not_called()
  trace = ctrl.last_dispatch_trace
  assert trace["source"] == "rule"
  print("  [OK] dispatch 付费事件走规则路由（不调用 LLM）")


# ================================================================
# LLMController.dispatch — 专家组路径
# ================================================================

def _make_mock_model(*expert_outputs: str):
  """创建按顺序返回多个 JSON 的 mock model。"""
  model = MagicMock()
  results = []
  for output in expert_outputs:
    result = MagicMock()
    result.content = output
    results.append(result)
  model.ainvoke = AsyncMock(side_effect=results)
  return model


def test_dispatch_ensemble_normal():
  """普通弹幕走专家组并行→合并。"""
  model = _make_mock_model(
    '{"should_reply": true, "urgency": 7, "has_action_request": false, "action_hint": ""}',
    '{"response_style": "detailed", "sentences": 2, "tone_hint": "认真"}',
    '{"memory_strategy": "normal", "session_mode": "comment_focus"}',
    '{"has_action_request": false, "action_hint": ""}',
  )
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你好", is_new=True),
    ),
    energy=0.8, patience=0.6,
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.should_reply is True
  assert plan.urgency == 7
  assert plan.response_style == "detailed"
  assert plan.sentences == 3
  assert plan.route_kind == "chat"
  assert plan.priority == 1
  trace = ctrl.last_dispatch_trace
  assert trace["source"] == "ensemble"
  assert "reply_judge" in trace["experts"]
  assert "style_advisor" in trace["experts"]
  assert "context_advisor" in trace["experts"]
  assert "action_guard" in trace["experts"]
  print("  [OK] dispatch 弹幕走专家组并行")


def test_dispatch_ensemble_with_enrichment():
  """专家组路径 + 规则增强字段正确合并。"""
  model = _make_mock_model(
    '{"should_reply": true, "urgency": 6, "has_action_request": false, "action_hint": ""}',
    '{"response_style": "existential", "sentences": 2}',
    '{"memory_strategy": "deep_recall", "session_mode": "comment_focus", '
    '"session_anchor": "LLM生成的锚点"}',
    '{"has_action_request": false, "action_hint": ""}',
  )
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你是AI吗？", is_new=True),
    ),
    available_persona_sections=("existential", "streaming"),
    available_knowledge_topics=("Neuro-sama",),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.route_kind == "chat"
  assert plan.sentences == 3
  assert "existential" in plan.persona_sections
  assert plan.session_anchor == "LLM生成的锚点"
  print("  [OK] dispatch 专家组 + 规则增强合并正确")


def test_dispatch_context_advisor_selects_corpus_tags():
  """ContextAdvisor 选择的 corpus tag 应透传到最终 plan。"""
  model = _make_mock_model(
    '{"should_reply": true, "urgency": 5, "has_action_request": false, "action_hint": ""}',
    '{"response_style": "normal", "sentences": 1}',
    '{"memory_strategy": "normal", "session_mode": "comment_focus", '
    '"corpus_style": "搞笑", "corpus_scene": "互动"}',
    '{"has_action_request": false, "action_hint": ""}',
  )
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你这波也太搞了", is_new=True),
    ),
    available_corpus_styles=("搞笑", "感性"),
    available_corpus_scenes=("互动", "冷场"),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.corpus_style == "搞笑"
  assert plan.corpus_scene == "互动"
  print("  [OK] dispatch: ContextAdvisor 的 corpus 标签可透传")


def test_context_advisor_filters_unknown_corpus_tags():
  """ContextAdvisor 只能返回 catalog 里存在的 corpus 标签。"""
  model = MagicMock()
  result = MagicMock()
  result.content = (
    '{"memory_strategy": "normal", "session_mode": "comment_focus", '
    '"corpus_style": "野生标签", "corpus_scene": "互动"}'
  )
  model.ainvoke = AsyncMock(return_value=result)
  advisor = ContextAdvisor(model=model, timeout=0.5)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="接个梗", is_new=True),
    ),
    available_corpus_styles=("搞笑",),
    available_corpus_scenes=("互动",),
  )
  from llm_controller.rule_router import RuleEnrichment

  outcome = asyncio.run(advisor.judge(inp, RuleEnrichment()))
  assert outcome.fields["corpus_style"] == ""
  assert outcome.fields["corpus_scene"] == "互动"
  print("  [OK] ContextAdvisor 会过滤非法 corpus 标签")


def test_reply_judge_parses_action_fields():
  """ReplyJudge 应解析 has_action_request/action_hint 字段。"""
  model = MagicMock()
  result = MagicMock()
  result.content = (
    '{"urgency":6,"has_action_request":true,"action_hint":"观众想让主播调音量"}'
  )
  model.ainvoke = AsyncMock(return_value=result)
  judge = ReplyJudge(model=model, timeout=0.5)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="声音大点", is_new=True),
    ),
  )
  from llm_controller.rule_router import RuleEnrichment

  outcome = asyncio.run(judge.judge(inp, RuleEnrichment()))
  assert outcome.fields["has_action_request"] is True
  assert "调音量" in outcome.fields["action_hint"]
  assert outcome.fields["urgency"] == 6
  print("  [OK] ReplyJudge 会解析动作护栏字段")


def test_dispatch_expert_timeout_uses_defaults():
  """单个专家超时时，用默认值补全，其他专家结果保留。"""
  model = MagicMock()
  call_count = 0

  async def selective_invoke(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      await asyncio.sleep(10)
    result = MagicMock()
    if call_count == 2:
      result.content = '{"response_style": "brief", "sentences": 1}'
    else:
      result.content = '{"memory_strategy": "normal", "session_mode": "comment_focus"}'
    return result

  model.ainvoke = selective_invoke
  ctrl = LLMController(model=model, timeout=0.1)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="测试", is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.should_reply is True
  assert plan.route_kind == "chat"
  print("  [OK] dispatch 单个专家超时，其余正常，合并成功")


def test_dispatch_force_fallback():
  """force_fallback 走纯规则 chat plan。"""
  model = MagicMock()
  model.ainvoke = AsyncMock(side_effect=RuntimeError("不应被调用"))
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你好", is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp, force_fallback=True))
  assert plan.should_reply is True
  assert plan.route_kind == "chat"
  assert plan.sentences == 2
  model.ainvoke.assert_not_called()
  trace = ctrl.last_dispatch_trace
  assert trace["source"] == "fallback_forced"
  print("  [OK] dispatch force_fallback 走纯规则 chat")


def test_dispatch_no_model():
  """无模型时走纯规则 chat plan。"""
  ctrl = LLMController(model=None, base_url="")
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="你好", is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  assert plan.should_reply is True
  assert plan.route_kind == "chat"
  assert plan.sentences == 2
  trace = ctrl.last_dispatch_trace
  assert trace["source"] == "rule_no_model"
  print("  [OK] dispatch 无模型走规则 chat")


# ================================================================
# 专家 JSON 容错
# ================================================================

def test_context_advisor_accepts_quoted_json_string():
  """ContextAdvisor 应能解析被字符串包裹的 JSON。"""
  from llm_controller.experts import ContextAdvisor
  from llm_controller.rule_router import RuleEnrichment

  result = MagicMock()
  result.content = '"{\\"memory_strategy\\":\\"normal\\",\\"session_mode\\":\\"comment_focus\\",\\"corpus_style\\":\\"\\",\\"corpus_scene\\":\\"\\"}"'
  model = MagicMock()
  model.ainvoke = AsyncMock(return_value=result)
  advisor = ContextAdvisor(model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="你好", is_new=True),
    ),
  )
  expert = asyncio.run(advisor.judge(inp, RuleEnrichment()))
  assert expert.source == "llm"
  assert expert.fields["memory_strategy"] == "normal"
  assert expert.fields["session_mode"] == "comment_focus"
  print("  [OK] ContextAdvisor 可解析字符串包裹的 JSON")


def test_action_guard_accepts_prefixed_json_and_content_blocks():
  """ActionGuard 应能解析带前缀或 content block 的 JSON。"""
  from llm_controller.experts import ActionGuard
  from llm_controller.rule_router import RuleEnrichment

  result = MagicMock()
  result.content = [
    {"type": "text", "text": '结果如下：{"has_action_request":true,"action_hint":"观众想让主播切歌"}'},
  ]
  model = MagicMock()
  model.ainvoke = AsyncMock(return_value=result)
  guard = ActionGuard(model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A", content="帮我切歌", is_new=True),
    ),
  )
  expert = asyncio.run(guard.judge(inp, RuleEnrichment()))
  assert expert.source == "llm"
  assert expert.fields["has_action_request"] is True
  assert "切歌" in expert.fields["action_hint"]
  print("  [OK] ActionGuard 可解析前缀文本与 content block JSON")


# ================================================================
# LLMController — per-expert model wiring
# ================================================================

def test_controller_accepts_distinct_expert_models():
  """构造器应把不同 expert model 分别绑定到对应专家。"""
  shared_model = MagicMock(name="shared_model")
  reply_model = MagicMock(name="reply_model")
  style_model = MagicMock(name="style_model")
  context_model = MagicMock(name="context_model")
  action_model = MagicMock(name="action_model")
  ctrl = LLMController(
    model=shared_model,
    model_name="per-expert",
    expert_models={
      "reply_judge": reply_model,
      "style_advisor": style_model,
      "context_advisor": context_model,
      "action_guard": action_model,
    },
  )
  assert ctrl._model is shared_model
  assert ctrl._reply_judge is not None and ctrl._reply_judge._model is reply_model
  assert ctrl._style_advisor is not None and ctrl._style_advisor._model is style_model
  assert ctrl._context_advisor is not None and ctrl._context_advisor._model is context_model
  assert ctrl._action_guard is not None and ctrl._action_guard._model is action_model
  print("  [OK] controller per-expert model wiring 生效")


def test_run_remote_default_wiring_uses_three_models():
  """run_remote 默认 wiring 应收敛到 3 路请求，ContextAdvisor 切到 gpt-5-mini。"""
  from langchain_wrapper import ModelType
  from run_remote import _build_controller_expert_models

  class FakeModelProvider:
    def __init__(self):
      self.calls = []

    def get_model(self, model_type, model_name=None, **kwargs):
      self.calls.append((model_type, model_name, kwargs))
      return {"model_type": model_type, "model_name": model_name, "kwargs": kwargs}

  provider = FakeModelProvider()
  _, model_name, expert_models, expert_labels = _build_controller_expert_models(provider)
  assert model_name == "per-expert"
  assert provider.calls[0][0] == ModelType.DEEPSEEK
  assert provider.calls[0][1] == "deepseek-chat"
  assert "deepseek" in expert_labels["reply_judge"]
  assert provider.calls[1][0] == ModelType.OPENAI
  assert provider.calls[1][1] == "gpt-5-mini"
  assert "ContextAdvisor: gpt-5-mini via openai" == expert_labels["context_advisor"]
  assert provider.calls[2][0] == ModelType.DEEPSEEK
  assert provider.calls[2][1] == "deepseek-chat"
  assert "StyleAdvisor: deepseek-chat via deepseek" == expert_labels["style_advisor"]
  assert len(provider.calls) == 3
  assert "action_guard" not in expert_labels
  assert expert_models["reply_judge"]["model_type"] == ModelType.DEEPSEEK
  assert expert_models["context_advisor"]["model_type"] == ModelType.OPENAI
  assert expert_models["style_advisor"]["model_type"] == ModelType.DEEPSEEK
  assert expert_models["action_guard"] is None
  print("  [OK] run_remote 默认 3 路 expert wiring，ActionGuard 并入 ReplyJudge")


# ================================================================
# ActionGuard — 动作请求注入 extra_instructions
# ================================================================

def test_dispatch_action_guard_injects_instruction():
  """ActionGuard 检测到动作请求时，extra_instructions 中包含提示。"""
  model = _make_mock_model(
    '{"should_reply": true, "urgency": 5, "has_action_request": true, "action_hint": "观众想让主播切歌"}',
    '{"response_style": "normal", "sentences": 1}',
    '{"memory_strategy": "normal", "session_mode": "comment_focus"}',
    '{"has_action_request": true, "action_hint": "观众想让主播切歌"}',
  )
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="帮我切首歌呗", is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  joined = " ".join(plan.extra_instructions)
  assert "无法执行" in joined
  assert "切歌" in joined
  assert "不要假装答应" in joined
  print("  [OK] ActionGuard 检测到动作请求 → extra_instructions 注入")


def test_dispatch_action_guard_no_action():
  """无动作请求时，ActionGuard 不注入任何内容。"""
  model = _make_mock_model(
    '{"should_reply": true, "urgency": 5, "has_action_request": false, "action_hint": ""}',
    '{"response_style": "normal", "sentences": 1}',
    '{"memory_strategy": "normal", "session_mode": "comment_focus"}',
    '{"has_action_request": false, "action_hint": ""}',
  )
  ctrl = LLMController(model=model)
  inp = ControllerInput(
    comments=(
      CommentBrief(id="1", user_id="u1", nickname="A",
                   content="今天天气不错", is_new=True),
    ),
  )
  plan = asyncio.run(ctrl.dispatch(inp))
  for instr in plan.extra_instructions:
    assert "无法执行" not in instr
  print("  [OK] ActionGuard 无动作请求 → 不注入")


# ================================================================
# _merge — 集成合并逻辑
# ================================================================

def test_merge_expert_overrides_enrichment():
  """专家输出优先于规则增强的 suggested 字段。"""
  from llm_controller.rule_router import RuleEnrichment

  enrichment = RuleEnrichment(
    suggested_session_anchor="规则生成的锚点",
    suggested_extra_instructions=("规则指令",),
    persona_sections=("existential",),
  )
  expert_results = {
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 8,
      "has_action_request": False, "action_hint": "",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "detailed", "sentences": 3}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "deep_recall",
      "session_anchor": "LLM锚点",
      "extra_instructions": ["LLM指令"],
    }),
    "action_guard": ExpertResult(name="action_guard", fields={
      "has_action_request": False, "action_hint": "",
    }),
  }
  plan = LLMController._merge(expert_results, enrichment)
  assert plan.session_anchor == "LLM锚点"
  assert plan.sentences == 4
  assert plan.extra_instructions == ("LLM指令",)
  assert plan.persona_sections == ("existential",)
  print("  [OK] merge: 专家输出优先于规则建议")


def test_merge_fallback_to_enrichment():
  """专家未给 session_anchor 时回退到规则增强的 suggested 值。"""
  from llm_controller.rule_router import RuleEnrichment

  enrichment = RuleEnrichment(
    suggested_session_anchor="规则生成的锚点",
    suggested_extra_instructions=("规则指令",),
  )
  expert_results = {
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 5,
      "has_action_request": False, "action_hint": "",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 2}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "normal",
      "session_anchor": "",
      "extra_instructions": [],
    }),
    "action_guard": ExpertResult(name="action_guard", fields={
      "has_action_request": False, "action_hint": "",
    }),
  }
  plan = LLMController._merge(expert_results, enrichment)
  assert plan.session_anchor == "规则生成的锚点"
  assert plan.sentences == 3
  assert plan.extra_instructions == ("规则指令",)
  print("  [OK] merge: 专家空值回退到规则建议")


def test_merge_reply_judge_false_still_replies():
  """即便 ReplyJudge 返回 should_reply=false，专家组路径也应继续回复。"""
  from llm_controller.rule_router import RuleEnrichment

  enrichment = RuleEnrichment()
  expert_results = {
    "reply_judge": ExpertResult(name="reply_judge", fields={"should_reply": False, "urgency": 2}),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 1}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "normal",
      "session_mode": "comment_focus",
    }),
  }
  plan = LLMController._merge(expert_results, enrichment)
  assert plan.should_reply is True
  assert plan.urgency == 2
  assert plan.sentences == 2
  print("  [OK] merge: ReplyJudge 只决定紧急程度，不再决定是否回复")


def test_merge_action_guard_appends_instruction():
  """ActionGuard 检测到动作请求时，指令追加到已有 extra_instructions 之后。"""
  from llm_controller.rule_router import RuleEnrichment

  enrichment = RuleEnrichment()
  expert_results = {
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 5,
      "has_action_request": True, "action_hint": "观众想让主播调音量",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 1}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "normal",
      "extra_instructions": ["已有指令"],
    }),
    "action_guard": ExpertResult(name="action_guard", fields={
      "has_action_request": True, "action_hint": "观众想让主播调音量",
    }),
  }
  plan = LLMController._merge(expert_results, enrichment)
  assert len(plan.extra_instructions) == 2
  assert plan.extra_instructions[0] == "已有指令"
  assert "调音量" in plan.extra_instructions[1]
  assert "无法执行" in plan.extra_instructions[1]
  print("  [OK] merge: ActionGuard 动作提示追加到 extra_instructions")


def test_merge_reply_judge_action_fields_without_action_guard():
  """未启用独立 ActionGuard 时，ReplyJudge 的动作字段也能注入护栏指令。"""
  from llm_controller.rule_router import RuleEnrichment

  enrichment = RuleEnrichment()
  expert_results = {
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 5,
      "has_action_request": True, "action_hint": "观众想让主播切画面",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 1}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "normal",
      "extra_instructions": ["已有指令"],
    }),
  }
  plan = LLMController._merge(expert_results, enrichment)
  assert len(plan.extra_instructions) == 2
  assert plan.extra_instructions[0] == "已有指令"
  assert "切画面" in plan.extra_instructions[1]
  assert "无法执行" in plan.extra_instructions[1]
  print("  [OK] merge: 无独立 ActionGuard 时 ReplyJudge 动作字段也能追加提示")


def test_merge_corpus_fields_passthrough():
  """ContextAdvisor 给出的 corpus 标签应进入最终 plan。"""
  from llm_controller.rule_router import RuleEnrichment

  enrichment = RuleEnrichment()
  expert_results = {
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 5,
      "has_action_request": False, "action_hint": "",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 1}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "normal",
      "corpus_style": "搞笑",
      "corpus_scene": "互动",
    }),
    "action_guard": ExpertResult(name="action_guard", fields={
      "has_action_request": False, "action_hint": "",
    }),
  }
  plan = LLMController._merge(expert_results, enrichment)
  assert plan.corpus_style == "搞笑"
  assert plan.corpus_scene == "互动"
  print("  [OK] merge: corpus 标签透传成功")


def test_merge_disables_corpus_for_knowledge_and_deep_recall():
  """知识命中或 deep_recall 场景应关闭 corpus 触发。"""
  from llm_controller.rule_router import RuleEnrichment

  knowledge_plan = LLMController._merge({
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 6,
      "has_action_request": False, "action_hint": "",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 2}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "normal",
      "corpus_style": "搞笑",
      "corpus_scene": "互动",
    }),
    "action_guard": ExpertResult(name="action_guard", fields={"has_action_request": False, "action_hint": ""}),
  }, RuleEnrichment(knowledge_hit=True))
  assert knowledge_plan.corpus_style == ""
  assert knowledge_plan.corpus_scene == ""

  recall_plan = LLMController._merge({
    "reply_judge": ExpertResult(name="reply_judge", fields={
      "should_reply": True, "urgency": 5,
      "has_action_request": False, "action_hint": "",
    }),
    "style_advisor": ExpertResult(name="style_advisor", fields={"response_style": "normal", "sentences": 1}),
    "context_advisor": ExpertResult(name="context_advisor", fields={
      "memory_strategy": "deep_recall",
      "corpus_style": "搞笑",
      "corpus_scene": "互动",
    }),
    "action_guard": ExpertResult(name="action_guard", fields={"has_action_request": False, "action_hint": ""}),
  }, RuleEnrichment())
  assert recall_plan.corpus_style == ""
  assert recall_plan.corpus_scene == ""
  print("  [OK] merge: 知识 / deep_recall 场景禁用 corpus")


# ================================================================
# Runner
# ================================================================

if __name__ == "__main__":
  tests = [
    ("PromptPlan.from_dict", [
      test_from_dict_normal,
      test_from_dict_defaults,
      test_from_dict_clamping,
      test_from_dict_route_priority_normalization,
    ]),
    ("CommentBrief.to_prompt_line", [
      test_comment_brief_danmaku,
      test_comment_brief_paid,
    ]),
    ("RuleRouter.route", [
      test_rule_paid,
      test_rule_guard,
      test_rule_normal_danmaku,
      test_rule_guard_entry,
      test_rule_entry_with_danmaku_goes_to_experts,
      test_rule_no_comments_short_silence,
      test_rule_no_comments_long_silence,
      test_rule_vlm_scene,
    ]),
    ("RuleRouter 增强信号", [
      test_enrichment_existential,
      test_enrichment_knowledge,
      test_enrichment_fake_gift,
      test_enrichment_relationship,
    ]),
    ("LLMController.dispatch", [
      test_dispatch_rule_route,
      test_dispatch_ensemble_normal,
      test_dispatch_ensemble_with_enrichment,
      test_dispatch_context_advisor_selects_corpus_tags,
      test_dispatch_expert_timeout_uses_defaults,
      test_dispatch_force_fallback,
      test_dispatch_no_model,
    ]),
    ("专家 JSON 容错", [
      test_context_advisor_accepts_quoted_json_string,
      test_action_guard_accepts_prefixed_json_and_content_blocks,
    ]),
    ("LLMController wiring", [
      test_controller_accepts_distinct_expert_models,
      test_run_remote_default_wiring_uses_three_models,
    ]),
    ("ContextAdvisor 语料触发", [
      test_context_advisor_filters_unknown_corpus_tags,
    ]),
    ("ReplyJudge 动作字段", [
      test_reply_judge_parses_action_fields,
    ]),
    ("ActionGuard 能力边界", [
      test_dispatch_action_guard_injects_instruction,
      test_dispatch_action_guard_no_action,
    ]),
    ("_merge 集成合并", [
      test_merge_expert_overrides_enrichment,
      test_merge_fallback_to_enrichment,
      test_merge_reply_judge_false_still_replies,
      test_merge_action_guard_appends_instruction,
      test_merge_reply_judge_action_fields_without_action_guard,
      test_merge_corpus_fields_passthrough,
      test_merge_disables_corpus_for_knowledge_and_deep_recall,
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
