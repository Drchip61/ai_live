"""
LLM Controller — 统一场景化调度核心

用本地 Qwen 3.5-9B 作为 Controller LLM，
将弹幕分类、回复决策、风格选择、记忆策略、会话控制、调度建议
全部合并到一次 Controller 调用中，输出结构化 PromptPlan。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import json_repair
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from prompts import PromptLoader
from .schema import ControllerInput, PromptPlan

logger = logging.getLogger(__name__)

_CONTROLLER_TIMEOUT = 3.0
_DEEP_NIGHT_PHASE_KEYWORDS = ("深夜", "夜聊", "夜深", "凌晨", "收尾", "晚安")
_IDENTITY_EXISTENTIAL_PATTERNS = (
  re.compile(r"你(?:到底)?是(?:不?是)?\s*(?:ai|人工智能|程序|模型|机器人)(?:吗|吧|呀|\?|？)?", re.I),
  re.compile(r"你(?:到底)?是(?:不?是)?\s*真人(?:吗|吧|呀|\?|？)?"),
  re.compile(r"你(?:到底)?是(?:不?是)?\s*虚拟(?:的|人|角色)?(?:吗|吧|呀|\?|？)?"),
  re.compile(r"你(?:到底)?是什么"),
  re.compile(r"你有自我意识吗"),
)
_SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
  "existential": ("真实", "存在", "记忆", "忘记", "遗忘", "孤独", "孤单", "感情", "你是谁", "你是什么"),
  "gaming_hardcore": ("只狼", "黑魂", "魂系", "法环", "艾尔登", "怪猎", "怪物猎人", "空洞骑士", "boss"),
  "gaming_suffering": ("jump king", "getting over it", "掘地求升", "壶男", "iwanna", "i wanna", "受苦", "坐牢"),
  "galgame": ("galgame", "视觉小说", "恋爱游戏", "clannad", "air", "key社", "素晴日", "剧情线"),
  "music": ("唱歌", "歌回", "翻唱", "练歌", "歌单", "音准", "声线", "歌词", "鸟之诗"),
  "streaming": ("直播", "直播间", "开播", "下播", "设备", "麦克风", "翻车", "弹幕"),
  "relationships": ("还记得", "上次", "老粉", "回来了", "一直看", "好久不见", "辛苦了", "陪了我"),
}
_KNOWLEDGE_KEYWORDS: dict[str, tuple[str, ...]] = {
  "Neuro-sama": ("neuro-sama", "neuro", "neurosama"),
  "木几萌": ("木几萌",),
}
_FAKE_GIFT_PATTERNS = (
  re.compile(r"(送|刷|给(?:你|主播)|来个).{0,8}(礼物|火箭|飞机|小花花|舰长|提督|总督|sc|super ?chat|醒目留言)", re.I),
  re.compile(r"(上|开)(个|一个)?(舰长|提督|总督)"),
  re.compile(r"(sc|super ?chat).{0,6}(块|元|￥|¥|\d)", re.I),
)


class LLMController:
  """统一调度器：输入元数据摘要，输出结构化 PromptPlan"""

  def __init__(
    self,
    model: Optional[BaseChatModel] = None,
    base_url: str = "http://localhost:2001/v1",
    model_name: str = "qwen3.5-9b",
    timeout: float = _CONTROLLER_TIMEOUT,
  ):
    if model is not None:
      self._model = model
    elif not str(base_url or "").strip():
      self._model = None
    else:
      from langchain_openai import ChatOpenAI
      self._model = ChatOpenAI(
        model=model_name,
        api_key="not-needed",
        base_url=base_url,
        temperature=0.3,
        max_tokens=512,
      )

    self._timeout = timeout
    self._prompt_template = PromptLoader().load("controller/dispatch.txt")
    self._last_plan: Optional[PromptPlan] = None

  @property
  def last_plan(self) -> Optional[PromptPlan]:
    return self._last_plan

  async def dispatch(self, ctrl_input: ControllerInput) -> PromptPlan:
    """核心调度：输入元数据 → 输出 PromptPlan JSON"""
    if self._model is None:
      plan = self._fallback(ctrl_input)
      self._last_plan = plan
      return plan

    prompt_text = self._render_prompt(ctrl_input)
    try:
      result = await asyncio.wait_for(
        self._model.ainvoke([HumanMessage(content=prompt_text)]),
        timeout=self._timeout,
      )
      plan = self._parse_plan(result.content)
      self._last_plan = plan
      return plan
    except asyncio.TimeoutError:
      logger.warning("Controller 调用超时 (%.1fs)，走 fallback", self._timeout)
      return self._fallback(ctrl_input)
    except Exception as e:
      logger.warning("Controller 调用失败: %s，走 fallback", e)
      return self._fallback(ctrl_input)

  def _render_prompt(self, ctrl_input: ControllerInput) -> str:
    """将 ControllerInput 填充到 prompt 模板"""
    comments = ctrl_input.comments
    new_count = sum(1 for c in comments if c.is_new)

    formatted_comments = "\n".join(
      c.to_prompt_line() for c in comments
    ) if comments else "(无弹幕)"

    formatted_viewers = "\n".join(
      v.to_prompt_line() for v in ctrl_input.viewer_briefs
    ) if ctrl_input.viewer_briefs else "(无活跃观众)"

    formatted_topics = "\n".join(
      t.to_prompt_line() for t in ctrl_input.active_topics
    ) if ctrl_input.active_topics else "(无活跃话题)"

    rate_str = f"{ctrl_input.comment_rate:.1f}" if ctrl_input.comment_rate >= 0 else "未知"

    return self._prompt_template.format(
      energy=f"{ctrl_input.energy:.2f}",
      patience=f"{ctrl_input.patience:.2f}",
      atmosphere=ctrl_input.atmosphere or "正常",
      stream_phase=ctrl_input.stream_phase,
      emotion=ctrl_input.emotion or "正常",
      round_count=ctrl_input.round_count,
      formatted_comments=formatted_comments,
      new_count=new_count,
      silence=f"{ctrl_input.silence_seconds:.0f}",
      rate=rate_str,
      formatted_viewers=formatted_viewers,
      formatted_topics=formatted_topics,
      available_persona_sections=", ".join(ctrl_input.available_persona_sections) or "无",
      available_knowledge_topics=", ".join(ctrl_input.available_knowledge_topics) or "无",
      available_corpus_styles=", ".join(ctrl_input.available_corpus_styles) or "无",
      available_corpus_scenes=", ".join(ctrl_input.available_corpus_scenes) or "无",
      mode="对话模式" if ctrl_input.is_conversation_mode else "视频模式",
      scene_change="是" if ctrl_input.has_scene_change else "否",
      scene_desc=ctrl_input.scene_description or "",
      last_style=ctrl_input.last_response_style,
      last_topic=ctrl_input.last_topic or "无",
    )

  @staticmethod
  def _parse_plan(raw: str) -> PromptPlan:
    """解析 Controller LLM 输出的 JSON 为 PromptPlan"""
    text = raw.strip()

    # Qwen thinking mode: 剥离 <think>...</think> 标签
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    if text.startswith("```"):
      lines = text.split("\n")
      lines = [l for l in lines if not l.strip().startswith("```")]
      text = "\n".join(lines)

    try:
      data = json.loads(text)
    except json.JSONDecodeError:
      data = json_repair.loads(text)

    if not isinstance(data, dict):
      raise ValueError(f"Controller 输出不是 JSON 对象: {type(data)}")
    return PromptPlan.from_dict(data)

  @staticmethod
  def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(keyword.lower() in lowered for keyword in keywords)

  @staticmethod
  def _is_identity_existential_question(text: str) -> bool:
    normalized = str(text or "").strip()
    return any(pattern.search(normalized) for pattern in _IDENTITY_EXISTENTIAL_PATTERNS)

  @classmethod
  def _detect_fake_gift_ids(cls, comments) -> tuple[str, ...]:
    fake_ids: list[str] = []
    for comment in comments:
      if comment.event_type != "danmaku":
        continue
      text = str(comment.content or "").strip()
      if not text:
        continue
      if any(pattern.search(text) for pattern in _FAKE_GIFT_PATTERNS):
        fake_ids.append(comment.id)
    return tuple(fake_ids)

  @classmethod
  def _pick_persona_sections(
    cls,
    ctrl_input: ControllerInput,
    comments,
    has_guard_member: bool,
  ) -> tuple[str, ...]:
    available = set(ctrl_input.available_persona_sections)
    if not available:
      return ()

    current_text = "\n".join(
      str(comment.content or "").strip()
      for comment in comments
      if str(comment.content or "").strip()
    )
    picks: list[str] = []

    def add(section: str) -> None:
      if section in available and section not in picks:
        picks.append(section)

    if (
      cls._is_identity_existential_question(current_text)
      or cls._contains_any(current_text, _SECTION_KEYWORDS["existential"])
    ):
      add("existential")

    if cls._contains_any(current_text, _SECTION_KEYWORDS["gaming_suffering"]):
      add("gaming_suffering")
    elif cls._contains_any(current_text, _SECTION_KEYWORDS["gaming_hardcore"]):
      add("gaming_hardcore")

    if cls._contains_any(current_text, _SECTION_KEYWORDS["galgame"]):
      add("galgame")
    if cls._contains_any(current_text, _SECTION_KEYWORDS["music"]):
      add("music")
    if cls._contains_any(current_text, _SECTION_KEYWORDS["streaming"]):
      add("streaming")

    relationship_signal = (
      has_guard_member
      or any(v.has_callbacks or v.has_open_threads or v.is_guard_member for v in ctrl_input.viewer_briefs)
      or cls._contains_any(current_text, _SECTION_KEYWORDS["relationships"])
    )
    if relationship_signal:
      add("relationships")

    return tuple(picks[:2])

  @classmethod
  def _pick_silence_persona_sections(cls, ctrl_input: ControllerInput) -> tuple[str, ...]:
    available = set(ctrl_input.available_persona_sections)
    if not available:
      return ()

    picks: list[str] = []
    phase_text = str(ctrl_input.stream_phase or "")
    is_deep_night = cls._contains_any(phase_text, _DEEP_NIGHT_PHASE_KEYWORDS)

    if is_deep_night and ctrl_input.silence_seconds >= 20 and "existential" in available:
      picks.append("existential")
    if ctrl_input.silence_seconds >= 15 and "streaming" in available:
      picks.append("streaming")

    return tuple(picks[:2])

  @classmethod
  def _pick_knowledge_topics(
    cls,
    ctrl_input: ControllerInput,
    comments,
  ) -> tuple[str, ...]:
    available = ctrl_input.available_knowledge_topics
    if not available:
      return ()

    current_text = "\n".join(
      str(comment.content or "").strip()
      for comment in comments
      if str(comment.content or "").strip()
    )
    lowered = current_text.lower()
    picks: list[str] = []

    for topic in available:
      aliases = _KNOWLEDGE_KEYWORDS.get(topic, (topic,))
      if any(alias.lower() in lowered for alias in aliases):
        picks.append(topic)

    return tuple(dict.fromkeys(picks))

  @staticmethod
  def _fallback(ctrl_input: ControllerInput) -> PromptPlan:
    """极简规则 fallback：模型不可用时保底"""
    current_comments = ctrl_input.new_comments or list(ctrl_input.comments)
    has_guard_member = any(c.is_guard_member for c in current_comments)
    fake_gift_ids = LLMController._detect_fake_gift_ids(current_comments)
    persona_sections = LLMController._pick_persona_sections(
      ctrl_input,
      current_comments,
      has_guard_member=has_guard_member,
    )
    knowledge_topics = LLMController._pick_knowledge_topics(ctrl_input, current_comments)
    high_value_sections = tuple(
      section for section in persona_sections
      if section in ("relationships", "streaming")
    )
    if "relationships" in ctrl_input.available_persona_sections and "relationships" not in high_value_sections:
      high_value_sections = high_value_sections + ("relationships",)
    if len(high_value_sections) > 1:
      high_value_sections = high_value_sections[:1]
    existential_trigger = "existential" in persona_sections

    if any(c.event_type == "guard_buy" for c in current_comments):
      return PromptPlan(
        should_reply=True,
        urgency=9,
        route_kind="guard_buy",
        response_style="guard_thanks",
        sentences=3,
        memory_strategy="normal",
        persona_sections=high_value_sections,
        session_mode="comment_focus",
        priority=0,
      )

    sc_comments = [c for c in current_comments if c.event_type == "super_chat"]
    if sc_comments:
      max_price = max(c.price for c in sc_comments)
      return PromptPlan(
        should_reply=True,
        urgency=9,
        route_kind="super_chat",
        response_style="detailed",
        sentences=3 if max_price >= 100 else 2,
        memory_strategy="deep_recall" if has_guard_member else "normal",
        persona_sections=high_value_sections,
        knowledge_topics=knowledge_topics,
        session_mode="comment_focus",
        priority=0,
      )

    gift_comments = [c for c in current_comments if c.event_type == "gift"]
    if gift_comments:
      max_price = max(c.price for c in gift_comments)
      return PromptPlan(
        should_reply=True,
        urgency=7 if max_price >= 10 else 4,
        route_kind="gift",
        response_style="brief" if max_price < 10 else "normal",
        sentences=2 if max_price >= 10 else 1,
        memory_strategy="minimal",
        session_mode="comment_focus",
        priority=0 if max_price >= 5 else 2,
      )

    if any(c.event_type == "entry" for c in current_comments):
      return PromptPlan(
        should_reply=True,
        urgency=3,
        route_kind="entry",
        response_style="brief",
        sentences=1,
        memory_strategy="minimal",
        session_mode="comment_focus",
        priority=2,
      )

    danmaku_comments = [c for c in current_comments if c.event_type == "danmaku"]
    if danmaku_comments:
      has_question = any(("?" in c.content or "？" in c.content) for c in danmaku_comments)
      return PromptPlan(
        should_reply=True,
        urgency=7 if has_guard_member else 5,
        route_kind="chat",
        response_style="existential" if existential_trigger else ("detailed" if has_question else "normal"),
        sentences=2 if (has_question or existential_trigger) else 1,
        memory_strategy="deep_recall" if has_guard_member else "normal",
        persona_sections=persona_sections,
        knowledge_topics=knowledge_topics,
        fake_gift_ids=fake_gift_ids,
        session_mode="comment_focus",
        priority=1,
      )

    if (
      not ctrl_input.is_conversation_mode
      and ctrl_input.scene_description
      and ctrl_input.silence_seconds >= 12
    ):
      return PromptPlan(
        should_reply=False,
        urgency=4,
        route_kind="vlm",
        response_style="brief",
        sentences=1,
        memory_strategy="minimal",
        session_mode="video_focus",
        priority=3,
        proactive_speak=True,
        proactive_reason="fallback_scene_reaction",
      )

    if ctrl_input.silence_seconds > 15:
      silence_sections = LLMController._pick_silence_persona_sections(ctrl_input)
      existential_silence = "existential" in silence_sections
      return PromptPlan(
        should_reply=False,
        urgency=4 if existential_silence else 3,
        route_kind="proactive",
        response_style="existential" if existential_silence else "brief",
        sentences=1,
        memory_strategy="normal" if silence_sections else "minimal",
        persona_sections=silence_sections,
        priority=3,
        proactive_speak=True,
        proactive_reason="fallback_deep_night_existential" if existential_silence else "fallback_silence",
      )

    return PromptPlan(
      should_reply=False,
      urgency=0,
      route_kind="chat",
      response_style="normal",
      sentences=1,
      memory_strategy="minimal",
      priority=1,
    )

  def debug_state(self) -> dict:
    """调试快照"""
    plan = self._last_plan
    return {
      "last_plan": {
        "should_reply": plan.should_reply,
        "urgency": plan.urgency,
        "route_kind": plan.route_kind,
        "response_style": plan.response_style,
        "sentences": plan.sentences,
        "memory_strategy": plan.memory_strategy,
        "persona_sections": list(plan.persona_sections),
        "knowledge_topics": list(plan.knowledge_topics),
        "corpus_style": plan.corpus_style,
        "proactive_speak": plan.proactive_speak,
        "priority": plan.priority,
      } if plan else None,
    }
