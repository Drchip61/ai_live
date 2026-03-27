"""
规则路由器

处理所有确定性场景（付费事件、入场、沉默），
并提供规则增强信号供专家组使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .schema import CommentBrief, ControllerInput, PromptPlan, ViewerBrief


# ------------------------------------------------------------------
# 常量 & 正则
# ------------------------------------------------------------------

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

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
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

KNOWLEDGE_KEYWORDS: dict[str, tuple[str, ...]] = {
  "Neuro-sama": ("neuro-sama", "neuro sama", "neuro", "neurosama"),
  "木几萌": ("木几萌",),
}

COMPETITOR_KNOWLEDGE_TOPICS = frozenset(("Neuro-sama", "木几萌"))

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

_PLAIN_GREETING_PATTERNS = (
  re.compile(r"^(你好|hello|hi|嗨|晚上好|早上好|下午好|主播好)$", re.I),
  re.compile(r"^(早呀|早安|午安|晚安)$", re.I),
)


def _bump_sentences(value: int, *, delta: int = 1, minimum: int = 1, maximum: int = 4) -> int:
  return max(minimum, min(maximum, int(value) + delta))


# ------------------------------------------------------------------
# 纯函数工具
# ------------------------------------------------------------------

def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
  lowered = str(text or "").lower()
  return any(kw.lower() in lowered for kw in keywords)


def looks_like_question(text: str) -> bool:
  normalized = str(text or "").strip()
  if not normalized:
    return False
  if "?" in normalized or "？" in normalized:
    return True
  lowered = normalized.lower()
  if any(kw in lowered for kw in _QUESTION_KEYWORDS):
    return True
  return normalized.endswith(("吗", "呢", "么", "嘛", "呀"))


def mentions_external_ai_topic(text: str) -> bool:
  lowered = str(text or "").lower()
  for aliases in KNOWLEDGE_KEYWORDS.values():
    if any(alias.lower() in lowered for alias in aliases):
      return True
  return False


def is_identity_existential_question(text: str) -> bool:
  normalized = str(text or "").strip()
  if not normalized:
    return False
  if mentions_external_ai_topic(normalized) and not any(
    marker in normalized for marker in ("你", "主播", "你这", "你们主播")
  ):
    return False
  return any(p.search(normalized) for p in _IDENTITY_EXISTENTIAL_PATTERNS)


def detect_fake_gift_ids(comments: list[CommentBrief]) -> tuple[str, ...]:
  fake_ids: list[str] = []
  for c in comments:
    if c.event_type != "danmaku":
      continue
    text = str(c.content or "").strip()
    if text and any(p.search(text) for p in _FAKE_GIFT_PATTERNS):
      fake_ids.append(c.id)
  return tuple(fake_ids)


def pick_persona_sections(
  ctrl_input: ControllerInput,
  comments: list[CommentBrief],
  *,
  has_guard_member: bool,
) -> tuple[str, ...]:
  available = set(ctrl_input.available_persona_sections)
  if not available:
    return ()

  current_text = "\n".join(
    str(c.content or "").strip()
    for c in comments
    if str(c.content or "").strip()
  )
  picks: list[str] = []

  def add(section: str) -> None:
    if section in available and section not in picks:
      picks.append(section)

  if (
    is_identity_existential_question(current_text)
    or contains_any(current_text, SECTION_KEYWORDS["existential"])
  ):
    add("existential")

  if contains_any(current_text, SECTION_KEYWORDS["gaming_suffering"]):
    add("gaming_suffering")
  elif contains_any(current_text, SECTION_KEYWORDS["gaming_hardcore"]):
    add("gaming_hardcore")

  if contains_any(current_text, SECTION_KEYWORDS["galgame"]):
    add("galgame")
  if contains_any(current_text, SECTION_KEYWORDS["music"]):
    add("music")
  if contains_any(current_text, SECTION_KEYWORDS["streaming"]):
    add("streaming")

  current_viewers = _current_viewer_briefs(ctrl_input, comments)
  relationship_signal = (
    contains_any(current_text, SECTION_KEYWORDS["relationships"])
    or (
      any(v.has_open_threads for v in current_viewers)
      and not _is_plain_greeting_turn(comments)
    )
  )
  if relationship_signal:
    add("relationships")

  return tuple(picks[:2])


def pick_silence_persona_sections(ctrl_input: ControllerInput) -> tuple[str, ...]:
  available = set(ctrl_input.available_persona_sections)
  if not available:
    return ()

  picks: list[str] = []
  phase_text = str(ctrl_input.stream_phase or "")
  is_deep_night = contains_any(phase_text, _DEEP_NIGHT_PHASE_KEYWORDS)

  if is_deep_night and ctrl_input.silence_seconds >= 20 and "existential" in available:
    picks.append("existential")
  if ctrl_input.silence_seconds >= 15 and "streaming" in available:
    picks.append("streaming")

  return tuple(picks[:2])


def pick_knowledge_topics(
  ctrl_input: ControllerInput,
  comments: list[CommentBrief],
) -> tuple[str, ...]:
  available = ctrl_input.available_knowledge_topics
  if not available:
    return ()

  current_text = "\n".join(
    str(c.content or "").strip()
    for c in comments
    if str(c.content or "").strip()
  )
  lowered = current_text.lower()
  picks: list[str] = []

  for topic in available:
    aliases = KNOWLEDGE_KEYWORDS.get(topic, (topic,))
    if any(alias.lower() in lowered for alias in aliases):
      picks.append(topic)

  return tuple(dict.fromkeys(picks))


def _collect_current_text(comments: list[CommentBrief]) -> str:
  return "\n".join(
    str(c.content or "").strip()
    for c in comments
    if str(c.content or "").strip()
  )


def _current_viewer_briefs(
  ctrl_input: ControllerInput,
  comments: list[CommentBrief],
) -> list[ViewerBrief]:
  current_viewer_ids = [
    viewer_id for viewer_id in dict.fromkeys(
      str(comment.user_id or "").strip()
      for comment in comments
      if str(comment.user_id or "").strip()
    )
  ]
  if not current_viewer_ids:
    return list(ctrl_input.viewer_briefs)
  viewer_map = {
    str(viewer.viewer_id).strip(): viewer
    for viewer in ctrl_input.viewer_briefs
    if str(viewer.viewer_id).strip()
  }
  return [
    viewer_map[viewer_id]
    for viewer_id in current_viewer_ids
    if viewer_id in viewer_map
  ]


def _is_plain_greeting_turn(comments: list[CommentBrief]) -> bool:
  payloads = [
    str(comment.content or "").strip()
    for comment in comments
    if str(comment.content or "").strip()
  ]
  if not payloads:
    return False
  return all(
    any(pattern.fullmatch(payload) for pattern in _PLAIN_GREETING_PATTERNS)
    for payload in payloads
  )


# ------------------------------------------------------------------
# RuleEnrichment — 规则层分析结果
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RuleEnrichment:
  """规则层分析结果，供专家组和集成器使用。"""

  # 确定性字段（直接合并到最终 plan）
  persona_sections: tuple[str, ...] = ()
  knowledge_topics: tuple[str, ...] = ()
  fake_gift_ids: tuple[str, ...] = ()
  viewer_focus_ids: tuple[str, ...] = ()
  high_value_sections: tuple[str, ...] = ()

  # 分析信号（传递给专家 prompt 辅助判断）
  has_guard_member: bool = False
  existential_trigger: bool = False
  knowledge_hit: bool = False
  competitor_hit: bool = False
  relationship_signal: bool = False
  has_question: bool = False

  # 规则生成的会话建议（专家失败时的 fallback 默认值）
  suggested_session_anchor: str = ""
  suggested_extra_instructions: tuple[str, ...] = ()


# ------------------------------------------------------------------
# RuleRouter
# ------------------------------------------------------------------

class RuleRouter:
  """
  规则路由器：

  1. route()  — 尝试用规则处理确定性场景，不能决定则返回 None。
  2. 返回的 RuleEnrichment 同时供专家组和集成器使用。
  """

  def route(
    self,
    ctrl_input: ControllerInput,
  ) -> tuple[Optional[PromptPlan], RuleEnrichment]:
    """
    Returns:
      (plan, enrichment) — plan 为 None 表示需要专家组介入。
    """
    current_comments = ctrl_input.new_comments or list(ctrl_input.comments)
    enrichment = self._analyze(ctrl_input, current_comments)
    plan = self._try_route(ctrl_input, current_comments, enrichment)
    return plan, enrichment

  # ----------------------------------------------------------------
  # 核心分析
  # ----------------------------------------------------------------

  @staticmethod
  def _analyze(
    ctrl_input: ControllerInput,
    current_comments: list[CommentBrief],
  ) -> RuleEnrichment:
    has_guard_member = any(c.is_guard_member for c in current_comments)
    current_text = _collect_current_text(current_comments)
    current_viewers = _current_viewer_briefs(ctrl_input, current_comments)
    explicit_relationship = contains_any(current_text, SECTION_KEYWORDS["relationships"])
    plain_greeting = _is_plain_greeting_turn(current_comments)

    fake_gift_ids = detect_fake_gift_ids(current_comments)
    persona_sections = pick_persona_sections(
      ctrl_input, current_comments, has_guard_member=has_guard_member,
    )
    knowledge_topics = pick_knowledge_topics(ctrl_input, current_comments)
    competitor_topics = tuple(
      t for t in knowledge_topics if t in COMPETITOR_KNOWLEDGE_TOPICS
    )

    high_value_sections = tuple(
      s for s in persona_sections if s in ("relationships", "streaming")
    )
    if (
      "relationships" in ctrl_input.available_persona_sections
      and "relationships" not in high_value_sections
    ):
      high_value_sections = high_value_sections + ("relationships",)
    if len(high_value_sections) > 1:
      high_value_sections = high_value_sections[:1]

    existential_trigger = "existential" in persona_sections

    relationship_viewers = (
      list(current_viewers)
      if explicit_relationship
      else (
        []
        if plain_greeting
        else [v for v in current_viewers if v.has_open_threads]
      )
    )
    relationship_signal = (
      explicit_relationship
      or bool(relationship_viewers)
    )
    viewer_focus_ids = tuple(v.viewer_id for v in relationship_viewers[:1])

    danmaku_comments = [c for c in current_comments if c.event_type == "danmaku"]
    has_question = any(looks_like_question(c.content) for c in danmaku_comments)
    knowledge_hit = bool(knowledge_topics)
    competitor_hit = bool(competitor_topics)

    session_anchor, extra_instructions = _build_session_hints(
      relationship_signal=relationship_signal,
      relationship_viewers=relationship_viewers,
      knowledge_hit=knowledge_hit,
      competitor_hit=competitor_hit,
      knowledge_topics=knowledge_topics,
    )

    return RuleEnrichment(
      persona_sections=persona_sections,
      knowledge_topics=knowledge_topics,
      fake_gift_ids=fake_gift_ids,
      viewer_focus_ids=viewer_focus_ids,
      high_value_sections=high_value_sections,
      has_guard_member=has_guard_member,
      existential_trigger=existential_trigger,
      knowledge_hit=knowledge_hit,
      competitor_hit=competitor_hit,
      relationship_signal=relationship_signal,
      has_question=has_question,
      suggested_session_anchor=session_anchor,
      suggested_extra_instructions=extra_instructions,
    )

  # ----------------------------------------------------------------
  # 规则路由
  # ----------------------------------------------------------------

  @staticmethod
  def _try_route(
    ctrl_input: ControllerInput,
    current_comments: list[CommentBrief],
    enrichment: RuleEnrichment,
  ) -> Optional[PromptPlan]:

    # — guard_buy —
    if any(c.event_type == "guard_buy" for c in current_comments):
      return PromptPlan(
        should_reply=True, urgency=9, route_kind="guard_buy",
        response_style="guard_thanks", sentences=_bump_sentences(3),
        memory_strategy="normal",
        persona_sections=enrichment.high_value_sections,
        session_mode="comment_focus", priority=0,
      )

    # — super_chat —
    sc = [c for c in current_comments if c.event_type == "super_chat"]
    if sc:
      max_price = max(c.price for c in sc)
      return PromptPlan(
        should_reply=True, urgency=9, route_kind="super_chat",
        response_style="detailed",
        sentences=_bump_sentences(3 if max_price >= 100 else 2),
        memory_strategy="deep_recall" if enrichment.has_guard_member else "normal",
        persona_sections=enrichment.high_value_sections,
        knowledge_topics=enrichment.knowledge_topics,
        session_mode="comment_focus", priority=0,
      )

    # — gift —
    gifts = [c for c in current_comments if c.event_type == "gift"]
    if gifts:
      max_price = max(c.price for c in gifts)
      return PromptPlan(
        should_reply=True,
        urgency=7 if max_price >= 10 else 4,
        route_kind="gift",
        response_style="brief" if max_price < 10 else "normal",
        sentences=_bump_sentences(2 if max_price >= 10 else 1),
        memory_strategy="minimal",
        session_mode="comment_focus",
        priority=0 if max_price >= 5 else 3,
      )

    # — entry only（无弹幕同场）—
    danmaku = [c for c in current_comments if c.event_type == "danmaku"]
    entries = [c for c in current_comments if c.event_type == "entry"]
    if entries and not danmaku:
      has_guard_entry = any(c.is_guard_member for c in entries)
      if has_guard_entry:
        return PromptPlan(
          should_reply=True, urgency=6, route_kind="entry",
          response_style="normal", sentences=_bump_sentences(2),
          memory_strategy="normal",
          persona_sections=enrichment.high_value_sections,
          session_mode="comment_focus", priority=3,
          extra_instructions=(
            "这是会员进场欢迎，要比普通入场更热情，点名并点出等级，但不要误说成新上舰。",
          ),
        )
      return PromptPlan(
        should_reply=True, urgency=3, route_kind="entry",
        response_style="brief", sentences=_bump_sentences(1),
        memory_strategy="minimal",
        session_mode="comment_focus", priority=3,
      )

    # — 有弹幕 → 交给专家组 —
    if danmaku:
      return None

    # — 无弹幕 + 有画面 + 长沉默 → VLM —
    if (
      not ctrl_input.is_conversation_mode
      and ctrl_input.scene_description
      and ctrl_input.silence_seconds >= 12
    ):
      return PromptPlan(
        should_reply=False, urgency=4, route_kind="vlm",
        response_style="brief", sentences=_bump_sentences(1),
        memory_strategy="minimal",
        session_mode="video_focus", priority=4,
        proactive_speak=True, proactive_reason="rule_scene_reaction",
      )

    # — 无弹幕 + 长沉默 → 主动发言 —
    if ctrl_input.silence_seconds > 15:
      silence_sections = pick_silence_persona_sections(ctrl_input)
      existential_silence = "existential" in silence_sections
      return PromptPlan(
        should_reply=False,
        urgency=4 if existential_silence else 3,
        route_kind="proactive",
        response_style="existential" if existential_silence else "brief",
        sentences=_bump_sentences(1),
        memory_strategy="normal" if silence_sections else "minimal",
        persona_sections=silence_sections,
        priority=4, proactive_speak=True,
        proactive_reason="rule_deep_night_existential" if existential_silence else "rule_silence",
      )

    # — 默认：跳过 —
    return PromptPlan(
      should_reply=False, urgency=0, route_kind="chat",
      response_style="normal", sentences=_bump_sentences(1),
      memory_strategy="minimal", priority=1,
    )


# ------------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------------

def _build_session_hints(
  *,
  relationship_signal: bool,
  relationship_viewers: list,
  knowledge_hit: bool,
  competitor_hit: bool,
  knowledge_topics: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
  """生成规则级的会话建议，供专家失败时回退使用。"""
  session_anchor = ""
  extra: list[str] = []

  if relationship_signal:
    if relationship_viewers:
      primary = relationship_viewers[0]
      nickname = primary.nickname or primary.viewer_id
      if primary.has_open_threads:
        session_anchor = f"继续 {nickname} 上次没聊完的话头"
      else:
        session_anchor = f"接住 {nickname} 的关系牌或历史梗"
      if primary.last_topic:
        session_anchor += f"（上次话题：{primary.last_topic}）"
    else:
      session_anchor = "接住关系牌并继续聊下去"
    extra.append(
      "先接住这位观众的关系牌或上次没聊完的话头，再顺势往下聊；结尾可以轻轻追问一句。"
    )
  elif knowledge_hit and knowledge_topics:
    session_anchor = f"继续聊 {knowledge_topics[0]}"
    if competitor_hit:
      extra.append(
        "这是竞品话题，第一句先亮态度，别中立复述；可以毒舌一点，直接点对方的槽点、翻车点或短板。"
      )
      extra.append(
        "锐评必须基于已给出的知识事实，别编黑料；第二句再补事实或比较判断。"
      )
    else:
      extra.append(
        "这是观众认真提到的知识话题，先给出明确态度，再顺手补一句轻追问。"
      )

  return session_anchor, tuple(extra[:3])
