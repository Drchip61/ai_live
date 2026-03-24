"""
LLM Controller — 统一场景化调度核心

将弹幕分类、回复决策、风格选择、记忆策略、会话控制、调度建议
合并到一次 Controller 调用中，输出结构化 PromptPlan。
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import logging
import re
import time
from typing import Any, Optional

import json_repair
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from prompts import PromptLoader
from .schema import ControllerInput, PromptPlan

logger = logging.getLogger(__name__)

_CONTROLLER_TIMEOUT = 3.0
_RENDER_COMMENTS_LIMIT = 8
_RENDER_VIEWERS_LIMIT = 3
_RENDER_TOPICS_LIMIT = 4
_RENDER_RESOURCE_LIMIT = 6
_DEEP_NIGHT_PHASE_KEYWORDS = ("深夜", "夜聊", "夜深", "凌晨", "收尾", "晚安")
_IDENTITY_EXISTENTIAL_PATTERNS = (
  re.compile(r"你(?:到底)?是(?:不?是)?\s*(?:ai|人工智能|程序|模型|机器人)(?:吗|吧|呀|\?|？)?", re.I),
  re.compile(r"^(?:到底)?是不是\s*(?:ai|人工智能|程序|模型|机器人)(?:吗|吧|呀|\?|？)?$", re.I),
  re.compile(r"^(?:是|算)\s*(?:ai|人工智能|程序|模型|机器人)\s*吗(?:呀|呢)?$", re.I),
  re.compile(r"你(?:到底)?是(?:不?是)?\s*真人(?:吗|吧|呀|\?|？)?"),
  re.compile(r"你(?:到底)?是(?:不?是)?\s*虚拟(?:的|人|角色)?(?:吗|吧|呀|\?|？)?"),
  re.compile(r"^(?:你|主播)?\s*(?:到底)?是(?:真人|程序|ai|人工智能|模型|机器人)\s*还是\s*(?:真人|程序|ai|人工智能|模型|机器人)", re.I),
  re.compile(r"^(?:真人|程序|ai|人工智能|模型|机器人)\s*还是\s*(?:真人|程序|ai|人工智能|模型|机器人)$", re.I),
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
  "relationships": (
    "还记得", "上次", "老粉", "回来了", "一直看", "好久不见", "辛苦了", "陪了我",
    "认识我", "认得我", "还认得", "认出我", "记得我姓", "你答应过",
  ),
}
_KNOWLEDGE_KEYWORDS: dict[str, tuple[str, ...]] = {
  "Neuro-sama": ("neuro-sama", "neuro sama", "neuro", "neurosama"),
  "木几萌": ("木几萌",),
}
_COMPETITOR_KNOWLEDGE_TOPICS = frozenset(("Neuro-sama", "木几萌"))
_FAKE_GIFT_PATTERNS = (
  re.compile(r"(送|刷|给(?:你|主播)|来个).{0,8}(礼物|火箭|飞机|小花花|舰长|提督|总督|sc|super ?chat|醒目留言)", re.I),
  re.compile(r"(上|开)(个|一个)?(舰长|提督|总督)"),
  re.compile(r"(sc|super ?chat).{0,6}(块|元|￥|¥|\d)", re.I),
)
_QUESTION_KEYWORDS = (
  "是不是", "吗", "呢", "么", "为什么", "为啥", "怎么看", "咋看",
  "怎么说", "怎么理解", "知不知道", "能不能", "可不可以", "行不行",
  "会不会", "有没有", "算不算", "是什么", "啥意思",
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
    self._model_name = model_name
    self._prompt_template = PromptLoader().load("controller/dispatch.txt")
    self._last_plan: Optional[PromptPlan] = None
    self._last_dispatch_trace: Optional[dict[str, Any]] = None

  @property
  def last_plan(self) -> Optional[PromptPlan]:
    return self._last_plan

  @property
  def last_dispatch_trace(self) -> Optional[dict[str, Any]]:
    return deepcopy(self._last_dispatch_trace) if self._last_dispatch_trace else None

  def _remember_dispatch_trace(
    self,
    *,
    source: str,
    plan: PromptPlan,
    prompt_text: str,
    raw_output: str = "",
    error: str = "",
    latency_ms: float = 0.0,
  ) -> None:
    """记录最近一次 dispatch 的可观测信息，供回复落库与监控使用。"""
    self._last_dispatch_trace = {
      "source": source,
      "model_name": self._model_name,
      "timeout_s": self._timeout,
      "latency_ms": round(max(latency_ms, 0.0), 1),
      "prompt_chars": len(prompt_text),
      "raw_output": str(raw_output or ""),
      "plan_json": plan.to_dict(nested=False),
      "error": str(error or ""),
    }

  @staticmethod
  def _render_lines(lines, limit: int, empty_text: str) -> str:
    rendered = [
      str(line).strip()
      for line in lines
      if str(line).strip()
    ]
    if not rendered:
      return empty_text
    visible = rendered[:limit]
    remaining = len(rendered) - len(visible)
    if remaining > 0:
      visible.append(f"... 另有{remaining}项未展开")
    return "\n".join(visible)

  @staticmethod
  def _render_resource_list(items, limit: int) -> str:
    rendered = [
      str(item).strip()
      for item in items
      if str(item).strip()
    ]
    if not rendered:
      return "无"
    visible = rendered[:limit]
    remaining = len(rendered) - len(visible)
    if remaining > 0:
      visible.append(f"等{remaining}项")
    return ", ".join(visible)

  async def dispatch(
    self,
    ctrl_input: ControllerInput,
    *,
    force_fallback: bool = False,
    fallback_source: str = "fallback_forced",
  ) -> PromptPlan:
    """核心调度：输入元数据 → 输出 PromptPlan JSON"""
    prompt_text = self._render_prompt(ctrl_input)
    if force_fallback or self._model is None:
      plan = self._postprocess_plan(self._fallback(ctrl_input), ctrl_input)
      self._last_plan = plan
      self._remember_dispatch_trace(
        source=fallback_source if force_fallback else "fallback_no_model",
        plan=plan,
        prompt_text=prompt_text,
      )
      return plan

    started = time.monotonic()
    try:
      result = await asyncio.wait_for(
        self._model.ainvoke([HumanMessage(content=prompt_text)]),
        timeout=self._timeout,
      )
      latency_ms = (time.monotonic() - started) * 1000
      raw_output = str(getattr(result, "content", "") or "")
      plan = self._postprocess_plan(self._parse_plan(raw_output), ctrl_input)
      self._last_plan = plan
      self._remember_dispatch_trace(
        source="llm",
        plan=plan,
        prompt_text=prompt_text,
        raw_output=raw_output,
        latency_ms=latency_ms,
      )
      return plan
    except asyncio.TimeoutError:
      logger.warning("Controller 调用超时 (%.1fs)，走 fallback", self._timeout)
      plan = self._postprocess_plan(self._fallback(ctrl_input), ctrl_input)
      self._last_plan = plan
      self._remember_dispatch_trace(
        source="fallback_timeout",
        plan=plan,
        prompt_text=prompt_text,
        error=f"timeout>{self._timeout:.1f}s",
        latency_ms=(time.monotonic() - started) * 1000,
      )
      return plan
    except Exception as e:
      logger.warning("Controller 调用失败: %s，走 fallback", e)
      plan = self._postprocess_plan(self._fallback(ctrl_input), ctrl_input)
      self._last_plan = plan
      self._remember_dispatch_trace(
        source="fallback_error",
        plan=plan,
        prompt_text=prompt_text,
        error=str(e),
        latency_ms=(time.monotonic() - started) * 1000,
      )
      return plan

  def _render_prompt(self, ctrl_input: ControllerInput) -> str:
    """将 ControllerInput 填充到 prompt 模板"""
    comments = ctrl_input.comments
    new_count = sum(1 for c in comments if c.is_new)

    formatted_comments = self._render_lines(
      (c.to_prompt_line() for c in comments),
      limit=_RENDER_COMMENTS_LIMIT,
      empty_text="(无弹幕)",
    )
    formatted_viewers = self._render_lines(
      (v.to_prompt_line() for v in ctrl_input.viewer_briefs),
      limit=_RENDER_VIEWERS_LIMIT,
      empty_text="(无活跃观众)",
    )
    formatted_topics = self._render_lines(
      (t.to_prompt_line() for t in ctrl_input.active_topics),
      limit=_RENDER_TOPICS_LIMIT,
      empty_text="(无活跃话题)",
    )

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
      available_persona_sections=self._render_resource_list(
        ctrl_input.available_persona_sections,
        limit=_RENDER_RESOURCE_LIMIT,
      ),
      available_knowledge_topics=self._render_resource_list(
        ctrl_input.available_knowledge_topics,
        limit=_RENDER_RESOURCE_LIMIT,
      ),
      available_corpus_styles=self._render_resource_list(
        ctrl_input.available_corpus_styles,
        limit=_RENDER_RESOURCE_LIMIT,
      ),
      available_corpus_scenes=self._render_resource_list(
        ctrl_input.available_corpus_scenes,
        limit=_RENDER_RESOURCE_LIMIT,
      ),
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
  def _looks_like_question(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
      return False
    if "?" in normalized or "？" in normalized:
      return True
    lowered = normalized.lower()
    if any(keyword in lowered for keyword in _QUESTION_KEYWORDS):
      return True
    return normalized.endswith(("吗", "呢", "么", "嘛", "呀"))

  @classmethod
  def _mentions_external_ai_topic(cls, text: str) -> bool:
    lowered = str(text or "").lower()
    for aliases in _KNOWLEDGE_KEYWORDS.values():
      if any(alias.lower() in lowered for alias in aliases):
        return True
    return False

  @classmethod
  def _is_identity_existential_question(cls, text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
      return False
    if cls._mentions_external_ai_topic(normalized) and not any(marker in normalized for marker in ("你", "主播", "你这", "你们主播")):
      return False
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

  @classmethod
  def _build_danmaku_only_input(
    cls,
    ctrl_input: ControllerInput,
  ) -> Optional[ControllerInput]:
    danmaku_comments = tuple(
      comment
      for comment in ctrl_input.new_comments
      if comment.event_type == "danmaku"
    )
    if not danmaku_comments:
      return None

    viewer_ids = {
      str(comment.user_id or "").strip()
      for comment in danmaku_comments
      if str(comment.user_id or "").strip()
    }
    viewer_briefs = tuple(
      viewer
      for viewer in ctrl_input.viewer_briefs
      if viewer.viewer_id in viewer_ids
    )
    return replace(
      ctrl_input,
      comments=danmaku_comments,
      viewer_briefs=viewer_briefs,
    )

  @classmethod
  def _postprocess_plan(
    cls,
    plan: PromptPlan,
    ctrl_input: ControllerInput,
  ) -> PromptPlan:
    danmaku_input = cls._build_danmaku_only_input(ctrl_input)
    if danmaku_input is None:
      return plan
    if plan.route_kind not in ("entry", "vlm", "proactive"):
      return plan

    chat_plan = cls._fallback(danmaku_input)
    return replace(
      chat_plan,
      topic_assignments=plan.topic_assignments or chat_plan.topic_assignments,
      corpus_style=plan.corpus_style or chat_plan.corpus_style,
      corpus_scene=plan.corpus_scene or chat_plan.corpus_scene,
      knowledge_topics=plan.knowledge_topics or chat_plan.knowledge_topics,
      persona_sections=plan.persona_sections or chat_plan.persona_sections,
      viewer_focus_ids=plan.viewer_focus_ids or chat_plan.viewer_focus_ids,
      extra_instructions=plan.extra_instructions or chat_plan.extra_instructions,
      tone_hint=plan.tone_hint or chat_plan.tone_hint,
      suggested_wait_min=plan.suggested_wait_min,
      suggested_wait_max=plan.suggested_wait_max,
    )

  @classmethod
  def _fallback(cls, ctrl_input: ControllerInput) -> PromptPlan:
    """极简规则 fallback：模型不可用时保底"""
    current_comments = ctrl_input.new_comments or list(ctrl_input.comments)
    has_guard_member = any(c.is_guard_member for c in current_comments)
    current_text = "\n".join(
      str(comment.content or "").strip()
      for comment in current_comments
      if str(comment.content or "").strip()
    )
    fake_gift_ids = cls._detect_fake_gift_ids(current_comments)
    persona_sections = cls._pick_persona_sections(
      ctrl_input,
      current_comments,
      has_guard_member=has_guard_member,
    )
    knowledge_topics = cls._pick_knowledge_topics(ctrl_input, current_comments)
    competitor_topics = tuple(
      topic for topic in knowledge_topics
      if topic in _COMPETITOR_KNOWLEDGE_TOPICS
    )
    high_value_sections = tuple(
      section for section in persona_sections
      if section in ("relationships", "streaming")
    )
    if "relationships" in ctrl_input.available_persona_sections and "relationships" not in high_value_sections:
      high_value_sections = high_value_sections + ("relationships",)
    if len(high_value_sections) > 1:
      high_value_sections = high_value_sections[:1]
    existential_trigger = "existential" in persona_sections
    relationship_viewers = [
      viewer for viewer in ctrl_input.viewer_briefs
      if viewer.has_callbacks or viewer.has_open_threads
    ]
    relationship_signal = (
      bool(relationship_viewers)
      or "relationships" in persona_sections
      or cls._contains_any(current_text, _SECTION_KEYWORDS["relationships"])
    )
    viewer_focus_ids = tuple(
      viewer.viewer_id for viewer in relationship_viewers[:1]
    )

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

    danmaku_comments = [c for c in current_comments if c.event_type == "danmaku"]
    entry_comments = [c for c in current_comments if c.event_type == "entry"]
    if entry_comments and not danmaku_comments:
      has_guard_entry = any(c.is_guard_member for c in entry_comments)
      if has_guard_entry:
        return PromptPlan(
          should_reply=True,
          urgency=6,
          route_kind="entry",
          response_style="normal",
          sentences=2,
          memory_strategy="normal",
          persona_sections=high_value_sections,
          session_mode="comment_focus",
          priority=2,
          extra_instructions=("这是会员进场欢迎，要比普通入场更热情，点名并点出等级，但不要误说成新上舰。",),
        )
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

    if danmaku_comments:
      has_question = any(cls._looks_like_question(c.content) for c in danmaku_comments)
      knowledge_hit = bool(knowledge_topics)
      competitor_hit = bool(competitor_topics)
      should_expand = has_question or existential_trigger or knowledge_hit or relationship_signal
      memory_strategy = "deep_recall" if (has_guard_member or relationship_signal) else "normal"
      extra_instructions: list[str] = []
      session_anchor = ""
      if relationship_signal:
        if relationship_viewers:
          primary_viewer = relationship_viewers[0]
          nickname = primary_viewer.nickname or primary_viewer.viewer_id
          if primary_viewer.has_open_threads:
            session_anchor = f"继续 {nickname} 上次没聊完的话头"
          else:
            session_anchor = f"接住 {nickname} 的关系牌或历史梗"
          if primary_viewer.last_topic:
            session_anchor += f"（上次话题：{primary_viewer.last_topic}）"
        else:
          session_anchor = "接住关系牌并继续聊下去"
        extra_instructions.append("先接住这位观众的关系牌或上次没聊完的话头，再顺势往下聊；结尾可以轻轻追问一句。")
      elif knowledge_hit:
        session_anchor = f"继续聊 {knowledge_topics[0]}"
        if competitor_hit:
          extra_instructions.append("这是竞品话题，第一句先亮态度，别中立复述；可以毒舌一点，直接点对方的槽点、翻车点或短板。")
          extra_instructions.append("锐评必须基于已给出的知识事实，别编黑料；第二句再补事实或比较判断。")
        else:
          extra_instructions.append("这是观众认真提到的知识话题，先给出明确态度，再顺手补一句轻追问。")
      return PromptPlan(
        should_reply=True,
        urgency=7 if has_guard_member else 5,
        route_kind="chat",
        response_style="existential" if existential_trigger else ("detailed" if should_expand else "normal"),
        sentences=2 if should_expand else 1,
        memory_strategy=memory_strategy,
        viewer_focus_ids=viewer_focus_ids,
        persona_sections=persona_sections,
        knowledge_topics=knowledge_topics,
        fake_gift_ids=fake_gift_ids,
        session_mode="comment_focus",
        session_anchor=session_anchor,
        priority=1,
        extra_instructions=tuple(extra_instructions[:3]),
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
      "last_plan": plan.to_dict() if plan else None,
      "last_dispatch_trace": self.last_dispatch_trace,
    }
