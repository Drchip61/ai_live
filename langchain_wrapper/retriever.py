"""
RetrieverResolver

从 Controller 决策中解析出本轮真正需要的上下文块，
并明确区分 trusted / untrusted 通道。
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, TYPE_CHECKING

from .contracts import ContextBlock, RetrievedContextBundle

if TYPE_CHECKING:
  from llm_controller.schema import PromptPlan
  from memory.manager import MemoryManager
  from emotion.state import EmotionMachine
  from emotion.affection import AffectionBank
  from meme.manager import MemeManager
  from style_bank import StyleBank
  from broadcaster_state import StateCard


def _event_type_str(comment) -> str:
  event_type = getattr(comment, "event_type", None)
  if hasattr(event_type, "value"):
    return str(event_type.value)
  return str(event_type or "danmaku")


_QUERY_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_QUERY_SPACE_RE = re.compile(r"\s+")
_QUERY_NOISE_PATTERNS = (
  re.compile(r"^(哈+|呵+|嘿+|6+|草+|啊+|嗯+|哦+|噢+|诶+|欸+|ww+|hhh+|233+|777+|[?？!！.。~～、，]+)$", re.I),
  re.compile(r"^(你好|hello|hi|嗨|晚上好|早上好|下午好|主播好)$", re.I),
)
_QUERY_CONTINUATION_PATTERNS = (
  re.compile(r"(上次|之前|刚才|继续|后来|后面|还记得|记不记得|不记得|你忘了|你不记得|我说了|我说过|我跟你说过|聊到哪|说到哪|上次说到哪|上次聊到哪|认得这个号|认出我|你说过|你提过|你答应)"),
  re.compile(r"^(那个|这个|那首|那部|那段|那局|那把|那位|那条)(呢|啊|呀|来着|怎么样|咋样|怎么说|后来呢)?$"),
)
_QUERY_RELATION_PATTERNS = (
  re.compile(r"(还记得我|记得我吗|我是谁|老粉|回来了|好久不见|上次聊|你说过|你答应|还记得这事|认识我不|你认识我不|认得我吗|还认得这个号吗|认出我没|还认得我吗|还记得我姓|记得我姓|不记得我|忘了我|你忘了吧)"),
)
_ALNUM_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+#-]{1,20}")
_CJK_QUERY_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")
_QUERY_CJK_PREFIXES_TO_STRIP = (
  "今天", "昨天", "刚才", "上次", "之前", "后来",
  "这个", "那个", "怎么", "为啥", "为什么",
  "那把", "这把", "那首", "这首", "那部", "这部",
  "那段", "这段", "那局", "这局", "那位", "这位",
  "那条", "这条",
)
_QUERY_CJK_EDGE_CHARS = frozenset("的了呢啊呀吧吗嘛和把这那就又还")
_QUERY_STOPWORDS = frozenset({
  "今天", "现在", "这个", "那个", "这里", "那里", "主播", "你们", "我们", "他们",
  "感觉", "就是", "还是", "然后", "因为", "所以", "怎么", "什么", "一下", "真的",
  "可以", "不是", "是不是", "有没有", "刚才", "继续", "上次", "之前", "后来",
  "记得", "回头", "那个呢", "这个呢", "主播好", "你好", "晚上好", "早上好", "下午好",
})


class RetrieverResolver:
  """把决策 plan 解析成结构化上下文 bundle。"""

  def __init__(
    self,
    *,
    memory_manager: Optional["MemoryManager"] = None,
    emotion_machine: Optional["EmotionMachine"] = None,
    affection_bank: Optional["AffectionBank"] = None,
    meme_manager: Optional["MemeManager"] = None,
    style_bank: Optional["StyleBank"] = None,
    state_card: Optional["StateCard"] = None,
  ) -> None:
    self._memory = memory_manager
    self._emotion = emotion_machine
    self._affection = affection_bank
    self._meme_manager = meme_manager
    self._style_bank = style_bank
    self._state_card = state_card

  async def resolve(
    self,
    plan: "PromptPlan",
    *,
    old_comments: list,
    new_comments: list,
    scene_context: str = "",
    viewer_ids: Optional[list[str]] = None,
    retrieval_query: str = "",
    writeback_input: str = "",
  ) -> RetrievedContextBundle:
    route_kind = getattr(plan, "route_kind", "chat")
    session_anchor = self._clean_query_text(
      str(getattr(plan, "session_anchor", "") or ""),
      max_len=80,
    )
    resolved_query = retrieval_query.strip() or self.build_retrieval_query(route_kind, old_comments, new_comments)
    if session_anchor:
      resolved_query = " ".join(
        part for part in (resolved_query, session_anchor) if part
      ).strip()
    resolved_writeback = writeback_input.strip() or self.build_writeback_input(
      route_kind,
      old_comments,
      new_comments,
      scene_context=scene_context,
    )
    current_comments = new_comments or old_comments[-2:]
    relationship_followup = any(
      self._looks_like_relationship_recall(self._comment_payload(comment))
      or self._looks_like_continuation(self._comment_payload(comment))
      for comment in current_comments
    )
    if self._anchor_needs_deep_recall(session_anchor):
      relationship_followup = True
    effective_memory_strategy = str(getattr(plan, "memory_strategy", "normal") or "normal")
    if relationship_followup and effective_memory_strategy in ("minimal", "normal"):
      effective_memory_strategy = "deep_recall"
    explicit_viewer_focus = tuple(plan.viewer_focus_ids)
    effective_viewer_ids = (
      explicit_viewer_focus
      or tuple(self._normalize_viewer_ids(viewer_ids))
      or self._collect_viewer_ids(old_comments + new_comments)
    )
    memory_viewer_ids = list(effective_viewer_ids)
    if (
      effective_memory_strategy == "normal"
      and not explicit_viewer_focus
      and memory_viewer_ids
    ):
      memory_viewer_ids = memory_viewer_ids[:1]

    blocks: list[ContextBlock] = []
    seen: set[tuple[str, str]] = set()

    def add_block(block: ContextBlock) -> None:
      rendered = block.render().strip()
      key = (block.trust, rendered)
      if not rendered or key in seen:
        return
      seen.add(key)
      blocks.append(block)

    if self._state_card is not None:
      add_block(ContextBlock(
        source="state_card",
        trust="trusted",
        text=self._state_card.to_prompt(),
      ))

    if self._emotion is not None:
      add_block(ContextBlock(
        source="emotion",
        trust="trusted",
        text=self._emotion.state.to_prompt(),
      ))

    if self._affection is not None:
      affection_text = self._affection.to_prompt()
      if affection_text:
        add_block(ContextBlock(
          source="affection",
          trust="trusted",
          text=affection_text,
        ))

    if session_anchor:
      add_block(ContextBlock(
        source="session_anchor",
        trust="trusted",
        title="【本轮提示】",
        text=f"- 优先延续：{session_anchor}",
      ))

    if self._memory is not None and effective_memory_strategy in ("normal", "deep_recall"):
      active_text, _, _ = await asyncio.to_thread(self._memory.retrieve_active_only)
      if active_text:
        add_block(ContextBlock(
          source="active_memory",
          trust="untrusted",
          text=active_text,
          query_used=resolved_query,
        ))

      structured_text = await asyncio.to_thread(
        self._memory.compile_structured_context,
        resolved_query,
        memory_viewer_ids,
        False,
        False,
        False,
        effective_memory_strategy,
      )
      if structured_text:
        add_block(ContextBlock(
          source="structured_memory",
          trust="untrusted",
          text=structured_text,
          query_used=resolved_query,
        ))

    if plan.persona_sections and self._memory is not None:
      persona_text = self._memory.get_persona_by_sections(list(plan.persona_sections))
      if persona_text:
        add_block(ContextBlock(
          source="persona_sections",
          trust="trusted",
          text=persona_text,
          title="【角色设定补充】",
        ))

    if plan.knowledge_topics and self._memory is not None:
      knowledge_text = self._memory.get_knowledge_by_topics(list(plan.knowledge_topics))
      if knowledge_text:
        add_block(ContextBlock(
          source="knowledge_topics",
          trust="trusted",
          text=knowledge_text,
          title="【参考知识】",
        ))

    if plan.corpus_style or plan.corpus_scene:
      query_used = resolved_query or "general"
      corpus_text = ""
      corpus_getter = getattr(self._memory, "get_corpus_context", None)
      if callable(corpus_getter):
        corpus_text = await asyncio.to_thread(
          corpus_getter,
          query_used,
          plan.corpus_style,
          plan.corpus_scene,
        )
      if corpus_text:
        add_block(ContextBlock(
          source="corpus_store",
          trust="trusted",
          title="【风格灵感】",
          text=corpus_text,
          query_used=query_used,
        ))
      elif self._style_bank is not None:
        style_text = await asyncio.to_thread(
          self._style_bank.retrieve_targeted,
          query=query_used,
          style_tag=plan.corpus_style,
          scene_tag=plan.corpus_scene,
        )
        if style_text:
          add_block(ContextBlock(
            source="style_bank",
            trust="trusted",
            text=style_text,
            query_used=query_used,
          ))

    if self._meme_manager is not None:
      meme_text = self._meme_manager.to_prompt()
      if meme_text:
        add_block(ContextBlock(
          source="meme_bank",
          trust="trusted",
          text=meme_text,
        ))

    if plan.extra_instructions:
      add_block(ContextBlock(
        source="extra_instructions",
        trust="trusted",
        title="【本轮提示】",
        text="\n".join(f"- {instruction}" for instruction in plan.extra_instructions),
      ))

    return RetrievedContextBundle(
      blocks=tuple(blocks),
      retrieval_query=resolved_query,
      writeback_input=resolved_writeback,
      viewer_ids=tuple(memory_viewer_ids),
    )

  @classmethod
  def build_retrieval_query(
    cls,
    route_kind: str,
    old_comments: list,
    new_comments: list,
  ) -> str:
    if route_kind not in ("chat", "super_chat"):
      return ""
    focus = new_comments or old_comments[-3:]
    current_comments = [
      comment for comment in focus[-5:]
      if cls._is_meaningful(comment)
    ]
    if not current_comments:
      return ""

    current_summaries = [cls._comment_summary(comment) for comment in current_comments[-2:]]
    keyword_hints: list[str] = []
    needs_history = False
    relationship_recall = False

    for comment in current_comments:
      payload = cls._clean_query_text(cls._comment_payload(comment), max_len=120)
      keywords = cls._extract_query_keywords(payload)
      keyword_hints.extend(keywords)
      if cls._needs_history_support(payload, keywords):
        needs_history = True
      if cls._looks_like_relationship_recall(payload):
        relationship_recall = True

    history_summaries: list[str] = []
    if needs_history:
      history_comments = [
        comment for comment in old_comments[-4:]
        if cls._is_meaningful(comment)
      ]
      history_summaries = [
        cls._comment_summary(comment) for comment in history_comments[-2:]
      ]
      for comment in history_comments[-2:]:
        keyword_hints.extend(
          cls._extract_query_keywords(cls._comment_payload(comment))
        )

    simple_query = (
      len(current_summaries) == 1
      and len(current_comments) == 1
      and not needs_history
      and not relationship_recall
    )
    if simple_query:
      return current_summaries[0]

    parts: list[str] = []
    if current_summaries:
      parts.append("当前弹幕 " + " ".join(current_summaries))
    if needs_history and history_summaries:
      parts.append("延续之前话题 " + " ".join(history_summaries))
    if relationship_recall:
      parts.append("关系回忆 上次互动 还记得")

    deduped_keywords = cls._dedupe_keep_order(keyword_hints)[:8]
    if deduped_keywords:
      parts.append("关键词 " + " ".join(deduped_keywords))
    return " ".join(part for part in parts if part).strip()

  @classmethod
  def build_writeback_input(
    cls,
    route_kind: str,
    old_comments: list,
    new_comments: list,
    *,
    scene_context: str = "",
  ) -> str:
    if route_kind in ("gift", "guard_buy", "entry"):
      return ""

    parts: list[str] = []
    focus = new_comments or old_comments[-3:]
    comment_lines = [
      f"观众「{getattr(comment, 'nickname', '')}」：{cls._comment_payload(comment)}"
      for comment in focus[-5:]
      if cls._is_meaningful(comment)
    ]
    if comment_lines:
      parts.append("；".join(comment_lines))

    if route_kind in ("vlm", "proactive"):
      scene_summary = cls._compact_scene_context(scene_context)
      if scene_summary:
        parts.insert(0, scene_summary)

    return "\n".join(part for part in parts if part).strip()

  @staticmethod
  def _compact_scene_context(scene_context: str) -> str:
    if not scene_context.strip():
      return ""
    lines = [line.strip() for line in scene_context.splitlines() if line.strip()]
    return " ".join(lines[:6])

  @classmethod
  def _collect_viewer_ids(cls, comments: list) -> tuple[str, ...]:
    result: list[str] = []
    for comment in comments:
      viewer_id = str(getattr(comment, "user_id", "") or "").strip()
      if viewer_id and viewer_id not in result:
        result.append(viewer_id)
    return tuple(result)

  @staticmethod
  def _normalize_viewer_ids(viewer_ids: Optional[list[str]]) -> list[str]:
    result: list[str] = []
    for viewer_id in viewer_ids or []:
      normalized = str(viewer_id or "").strip()
      if normalized and normalized not in result:
        result.append(normalized)
    return result

  @classmethod
  def _is_meaningful(cls, comment) -> bool:
    payload = cls._clean_query_text(cls._comment_payload(comment), max_len=80)
    if not payload:
      return False
    if any(pattern.fullmatch(payload) for pattern in _QUERY_NOISE_PATTERNS):
      return False
    if len(payload) > 2:
      return True
    return bool(cls._extract_query_keywords(payload, max_terms=1))

  @classmethod
  def _comment_summary(cls, comment) -> str:
    nickname = cls._clean_query_text(
      str(getattr(comment, "nickname", "") or "").strip(),
      max_len=40,
    )
    payload = cls._clean_query_text(cls._comment_payload(comment), max_len=120)
    return f"{nickname}：{payload}" if nickname else payload

  @staticmethod
  def _clean_query_text(text: str, max_len: int = 120) -> str:
    normalized = _QUERY_CONTROL_CHARS_RE.sub("", str(text or ""))
    normalized = normalized.replace("```", " ")
    normalized = _QUERY_SPACE_RE.sub(" ", normalized).strip()
    return normalized[:max_len]

  @classmethod
  def _extract_query_keywords(cls, text: str, max_terms: int = 6) -> list[str]:
    normalized = cls._clean_query_text(text, max_len=120)
    if not normalized:
      return []

    candidates: list[str] = []
    for token in _ALNUM_QUERY_TOKEN_RE.findall(normalized):
      clean = token.strip("._:+#-")
      if len(clean) >= 2:
        candidates.append(clean)

    for token in _CJK_QUERY_TOKEN_RE.findall(normalized):
      clean = cls._normalize_cjk_query_token(token)
      if len(clean) < 2 or clean in _QUERY_STOPWORDS:
        continue
      candidates.append(clean)

    return cls._dedupe_keep_order(candidates)[:max_terms]

  @classmethod
  def _needs_history_support(cls, payload: str, keywords: list[str]) -> bool:
    compact = cls._clean_query_text(payload, max_len=40)
    if cls._looks_like_continuation(compact):
      return True
    return len(compact) <= 6 and not keywords

  @staticmethod
  def _normalize_cjk_query_token(token: str) -> str:
    clean = str(token or "").strip()
    for prefix in _QUERY_CJK_PREFIXES_TO_STRIP:
      if clean.startswith(prefix) and len(clean) - len(prefix) >= 2:
        clean = clean[len(prefix):]
        break
    while len(clean) >= 2 and clean[:1] in _QUERY_CJK_EDGE_CHARS:
      clean = clean[1:]
    while len(clean) >= 2 and clean[-1:] in _QUERY_CJK_EDGE_CHARS:
      clean = clean[:-1]
    return clean

  @staticmethod
  def _dedupe_keep_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
      normalized = str(item or "").strip()
      if not normalized or normalized in seen:
        continue
      seen.add(normalized)
      result.append(normalized)
    return result

  @staticmethod
  def _looks_like_continuation(payload: str) -> bool:
    return any(pattern.search(payload) for pattern in _QUERY_CONTINUATION_PATTERNS)

  @staticmethod
  def _looks_like_relationship_recall(payload: str) -> bool:
    return any(pattern.search(payload) for pattern in _QUERY_RELATION_PATTERNS)

  @staticmethod
  def _anchor_needs_deep_recall(session_anchor: str) -> bool:
    anchor = str(session_anchor or "").strip()
    if not anchor:
      return False
    return any(marker in anchor for marker in (
      "关系", "回钩", "上次", "没聊完", "老粉", "继续聊",
    ))

  @staticmethod
  def _comment_payload(comment) -> str:
    event_type = _event_type_str(comment)
    if event_type == "guard_buy":
      level_map = {1: "舰长", 2: "提督", 3: "总督"}
      level = level_map.get(getattr(comment, "guard_level", 0), "舰长")
      return f"开通了{level}"
    if event_type == "super_chat":
      price = float(getattr(comment, "price", 0.0) or 0.0)
      content = str(getattr(comment, "content", "") or "").strip()
      return f"SC ¥{price:.0f}：{content}"
    if event_type == "gift":
      gift_name = str(getattr(comment, "gift_name", "") or "礼物")
      gift_num = getattr(comment, "gift_num", 0) or 1
      return f"赠送 {gift_name} x{gift_num}"
    if event_type == "entry":
      return "进入直播间"
    return str(getattr(comment, "content", "") or "").strip()
