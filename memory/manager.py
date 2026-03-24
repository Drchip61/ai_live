"""
记忆系统编排器

当前实现以 structured store 为主，legacy 向量层已从主链移除。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import json_repair
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EmbeddingConfig, MemoryConfig
from .context_store import (
  CorpusStore,
  ExternalKnowledgeStore,
  PersonaSpecStore,
  SelfMemoryStore,
  UserMemoryStore,
)
from .structured_retriever import StructuredMemoryRetriever
from .layers.active import ActiveLayer
from .prompts import (
  INTERACTION_SUMMARY_PROMPT,
  PERIODIC_SUMMARY_PROMPT,
  STANCE_EXTRACTION_PROMPT,
  VIEWER_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


class MemoryManager:
  """记忆系统顶层编排器（active + structured stores）"""

  _STANCE_INDICATORS = re.compile(
    r"我觉得|我认为|我喜欢|我讨厌|我比较|在我看来|我的看法|"
    r"我偏向|我支持|我反对|说实话我|我个人|依我看|"
    r"我更.{1,6}一些|我不太.{1,6}|我挺.{1,6}的|我是.{1,6}派|我站"
  )

  _NOISE_DANMAKU = re.compile(
    r"^(哈+|6+|[?？!！.。~～、，]+|草+|好家伙|啊+|呜+|嗯+|ww+|hhh+|lol+|emm+|"
    r"nb|tql|xswl|yyds|awsl|dd|ddd+|[Oo0]+|233+|7777*|牛|强|绝|顶|冲|来了|"
    r"你好|hello|hi|嗨|晚上好|早上好|下午好|主播好|"
    r"[👍👏🔥❤️💯😂🤣😭😍]+)$",
    re.IGNORECASE,
  )
  _GUARD_TERMS = (
    "舰长", "提督", "总督", "大航海会员",
    "舰长徽章", "提督徽章", "总督徽章", "徽章",
  )
  _GUARD_FACT_MARKERS = (
    "已经成为", "成为舰长", "成为提督", "成为总督",
    "是舰长", "是提督", "是总督",
    "拥有舰长徽章", "拥有提督徽章", "拥有总督徽章",
    "已挂舰长徽章", "已挂提督徽章", "已挂总督徽章",
    "开通了舰长", "开通了提督", "开通了总督",
    "开了舰长", "开了提督", "开了总督",
    "买了舰长", "买了提督", "买了总督",
    "确认了他的舰长身份", "确认了她的舰长身份", "确认了其舰长身份",
    "确认了他的提督身份", "确认了她的提督身份", "确认了其提督身份",
    "确认了他的总督身份", "确认了她的总督身份", "确认了其总督身份",
    "开通了大航海会员", "是大航海会员",
  )
  _GUARD_JOKE_MARKERS = (
    "嘴上", "玩笑", "梗", "调侃", "如果", "假设",
    "验证", "测试", "记得", "注意到", "问", "是不是",
    "会用", "话题", "互动",
  )

  def __init__(
    self,
    persona: str,
    config: MemoryConfig = MemoryConfig(),
    summary_model: Optional[BaseChatModel] = None,
    enable_global_memory: bool = True,
  ):
    self._persona = persona
    self._config = config
    self._enable_global_memory = enable_global_memory
    self._summary_model = summary_model
    self._session_id: Optional[str] = None

    if enable_global_memory:
      embedding_config = config.embedding
    else:
      embedding_config = EmbeddingConfig(
        model_name=config.embedding.model_name,
        persist_directory=None,
      )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    self._embeddings = HuggingFaceEmbeddings(
      model_name=embedding_config.model_name,
      model_kwargs={"device": device},
    )

    self._persona_static_dir = (
      Path(__file__).resolve().parent.parent
      / "personas"
      / persona
      / "static_memories"
    )

    self._structured_root: Optional[Path] = None
    if enable_global_memory and config.embedding.persist_directory:
      self._structured_root = Path(
        config.embedding.persist_directory
      ) / config.structured.directory_name

    user_memory_path = (
      self._structured_root / config.structured.user_memory_filename
      if self._structured_root is not None else None
    )
    self_memory_path = (
      self._structured_root / config.structured.self_memory_filename
      if self._structured_root is not None else None
    )
    persona_spec_path = (
      self._structured_root / config.structured.persona_spec_filename
      if self._structured_root is not None else None
    )
    corpus_path = (
      self._structured_root / config.structured.corpus_filename
      if self._structured_root is not None else None
    )
    knowledge_path = (
      self._structured_root / config.structured.external_knowledge_filename
      if self._structured_root is not None else None
    )

    self._user_memory_store: Optional[UserMemoryStore] = None
    self._self_memory_store: Optional[SelfMemoryStore] = None
    self._persona_spec_store: Optional[PersonaSpecStore] = None
    self._corpus_store: Optional[CorpusStore] = None
    self._external_knowledge_store: Optional[ExternalKnowledgeStore] = None
    self._structured_retriever: Optional[StructuredMemoryRetriever] = None

    if config.structured.enabled:
      self._user_memory_store = UserMemoryStore(user_memory_path)
      self._self_memory_store = SelfMemoryStore(self_memory_path)
      self._persona_spec_store = PersonaSpecStore(persona_spec_path, persona=persona)
      self._corpus_store = CorpusStore(corpus_path)
      self._external_knowledge_store = ExternalKnowledgeStore(knowledge_path)
      self._persona_spec_store.load_from_static_dir(self._persona_static_dir)
      self._structured_retriever = StructuredMemoryRetriever(
        user_memory_store=self._user_memory_store,
        self_memory_store=self._self_memory_store,
        persona_spec_store=self._persona_spec_store,
        corpus_store=self._corpus_store,
        external_knowledge_store=self._external_knowledge_store,
        embedding_config=embedding_config,
        embeddings=self._embeddings,
        config=config.structured,
      )
      self._structured_retriever.rebuild_all()
      self._structured_retriever.ensure_healthy()

    self._active = ActiveLayer(
      config=config.active,
      on_overflow=self._on_active_overflow,
    )

    self._summary_task: Optional[asyncio.Task] = None
    self._recent_interactions: list[tuple[str, str, datetime]] = []

  @property
  def user_memory_store(self) -> Optional[UserMemoryStore]:
    return self._user_memory_store

  @property
  def self_memory_store(self) -> Optional[SelfMemoryStore]:
    return self._self_memory_store

  @property
  def persona_spec_store(self) -> Optional[PersonaSpecStore]:
    return self._persona_spec_store

  @property
  def corpus_store(self) -> Optional[CorpusStore]:
    return self._corpus_store

  @property
  def external_knowledge_store(self) -> Optional[ExternalKnowledgeStore]:
    return self._external_knowledge_store

  @property
  def embeddings(self) -> HuggingFaceEmbeddings:
    return self._embeddings

  @property
  def session_id(self) -> Optional[str]:
    return self._session_id

  @session_id.setter
  def session_id(self, value: Optional[str]) -> None:
    self._session_id = value

  def _on_active_overflow(self, content: str, _timestamp: datetime, _response: str) -> None:
    """Active 层溢出后，把旧线头沉入 structured self threads。"""
    if self._self_memory_store is None:
      return
    self._self_memory_store.add_thread_memory(
      content,
      source_layer="active_overflow",
    )
    self._refresh_self_structured_indexes(include_threads=True)

  def _get_summary_model(self) -> BaseChatModel:
    if self._summary_model is None:
      from langchain_wrapper.model_provider import ModelProvider

      self._summary_model = ModelProvider.remote_small()
    return self._summary_model

  @classmethod
  def _contains_guard_claim(cls, text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
      return False
    if any(marker in normalized for marker in cls._GUARD_FACT_MARKERS):
      return True
    if any(term in normalized for term in cls._GUARD_TERMS):
      if "身份" in normalized or "待遇" in normalized:
        return True
      if normalized.startswith(("是", "已是", "已挂", "拥有", "确认")):
        return True
    return False

  @classmethod
  def _mentions_guard_topic(cls, text: str) -> bool:
    normalized = str(text or "").strip()
    return bool(normalized) and any(term in normalized for term in cls._GUARD_TERMS)

  @classmethod
  def _sanitize_guard_fact_entries(
    cls,
    items: list[dict],
    text_key: str,
  ) -> list[dict]:
    result: list[dict] = []
    for item in items:
      if not isinstance(item, dict):
        continue
      text = str(item.get(text_key, "")).strip()
      if not text or cls._contains_guard_claim(text):
        continue
      result.append(item)
    return result

  @classmethod
  def _sanitize_guard_topic_entries(
    cls,
    items: list[dict],
    source_mentions_guard_topic: bool = False,
  ) -> list[dict]:
    result: list[dict] = []
    for item in items:
      if not isinstance(item, dict):
        continue
      topic = str(item.get("topic", "")).strip()
      if topic and (
        any(term in topic for term in cls._GUARD_TERMS) or
        (source_mentions_guard_topic and "待遇" in topic)
      ):
        continue
      result.append(item)
    return result

  @classmethod
  def _sanitize_guard_callback_entries(cls, items: list[dict]) -> list[dict]:
    result: list[dict] = []
    for item in items:
      if not isinstance(item, dict):
        continue
      hook = str(item.get("hook", "")).strip()
      if not hook:
        continue
      normalized_hook = hook
      if cls._mentions_guard_topic(hook):
        if any(marker in hook for marker in cls._GUARD_JOKE_MARKERS):
          if "如果" in hook or "假设" in hook:
            normalized_hook = "会用“开舰长”这类假设性提问"
          elif any(marker in hook for marker in ("验证", "测试", "记得", "注意到", "徽章确认")):
            normalized_hook = "会拿舰长/徽章话题来验证主播记忆"
          else:
            normalized_hook = hook
        elif cls._contains_guard_claim(hook):
          continue
        else:
          normalized_hook = "会拿舰长/徽章话题和主播互动"
      cleaned = dict(item)
      cleaned["hook"] = normalized_hook.strip()
      if cleaned["hook"]:
        result.append(cleaned)
    return result

  @classmethod
  def _sanitize_guard_thread_entries(
    cls,
    items: list[dict],
    source_mentions_guard_topic: bool = False,
  ) -> list[dict]:
    result: list[dict] = []
    for item in items:
      if not isinstance(item, dict):
        continue
      thread = str(item.get("thread", "")).strip()
      if not thread:
        continue
      if cls._contains_guard_claim(thread):
        continue
      if source_mentions_guard_topic and any(marker in thread for marker in ("待遇", "徽章", "身份")):
        continue
      result.append(item)
    return result

  @classmethod
  def _sanitize_guard_relationship_state(
    cls,
    relationship_state: Optional[dict],
    source_mentions_guard_topic: bool = False,
  ) -> Optional[dict]:
    if not isinstance(relationship_state, dict):
      return relationship_state
    cleaned = dict(relationship_state)
    last_dialogue_stop = str(cleaned.get("last_dialogue_stop", "")).strip()
    if cls._contains_guard_claim(last_dialogue_stop):
      cleaned.pop("last_dialogue_stop", None)
      return cleaned
    if source_mentions_guard_topic and any(marker in last_dialogue_stop for marker in ("待遇", "徽章", "身份")):
      cleaned.pop("last_dialogue_stop", None)
    return cleaned

  def list_persona_sections(self) -> list[str]:
    if self._persona_spec_store is None:
      return []
    return self._persona_spec_store.list_sections()

  def list_knowledge_topics(self) -> list[str]:
    if self._external_knowledge_store is None:
      return []
    return self._external_knowledge_store.list_topics()

  def list_corpus_style_tags(self) -> list[str]:
    if self._corpus_store is None:
      return []
    return self._corpus_store.list_style_tags()

  def list_corpus_scene_tags(self) -> list[str]:
    if self._corpus_store is None:
      return []
    return self._corpus_store.list_scene_tags()

  def get_persona_by_sections(self, sections: list[str]) -> str:
    if not sections or self._persona_spec_store is None:
      return ""
    items = self._persona_spec_store.get_by_sections(sections)
    if not items:
      return ""
    lines = []
    limit = max(1, int(self._config.structured.persona_top_k or 4))
    for item in items[:limit]:
      section = str(item.get("section", "")).strip()
      text = str(item.get("text", "")).strip()
      if text:
        lines.append(f"{section}：{text}" if section else text)
    return "\n".join(lines)

  def get_knowledge_by_topics(self, topics: list[str]) -> str:
    if not topics or self._external_knowledge_store is None:
      return ""
    entries = self._external_knowledge_store.get_by_topics(topics)
    if not entries:
      return ""
    parts = []
    for entry in entries:
      head = entry.topic or entry.category
      text = f"【{head}】{entry.summary}"
      stance = str(entry.streamer_stance or "").strip()
      if stance:
        text += f"\n【主播立场】{stance}"
      usage_rules = [
        str(rule).strip()
        for rule in (entry.usage_rules or [])
        if str(rule).strip()
      ]
      if usage_rules:
        text += "\n【使用原则】\n" + "\n".join(
          f"- {rule}" for rule in usage_rules[:5]
        )
      if entry.facts:
        fact_lines: list[str] = []
        for fact in entry.facts[:5]:
          if not isinstance(fact, dict):
            fact_text = str(fact).strip()
            if fact_text:
              fact_lines.append(f"- {fact_text}")
            continue
          aspect = str(fact.get("aspect", "")).strip()
          content = str(fact.get("content", "")).strip()
          if aspect and content:
            fact_lines.append(f"- {aspect}：{content}")
          elif content:
            fact_lines.append(f"- {content}")
        if fact_lines:
          text += "\n【参考事实】\n" + "\n".join(fact_lines)
      parts.append(text)
    return "\n\n".join(parts)

  @staticmethod
  def _truncate_debug_snippet(value, max_chars: int = 240) -> str:
    try:
      if isinstance(value, str):
        text = value
      else:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
      text = repr(value)
    text = str(text or "").replace("\n", "\\n").strip()
    if len(text) <= max_chars:
      return text
    return text[:max_chars] + "..."

  @classmethod
  def _normalize_viewer_memory_items(
    cls,
    memories: list,
    raw_snippet: str = "",
  ) -> list[dict]:
    normalized: list[dict] = []
    skipped: list[str] = []
    for idx, item in enumerate(memories):
      if isinstance(item, dict):
        normalized.append(item)
        continue
      if isinstance(item, list):
        expanded = 0
        for nested_idx, nested in enumerate(item):
          if isinstance(nested, dict):
            normalized.append(nested)
            expanded += 1
          else:
            skipped.append(
              f"{idx}[{nested_idx}]={type(nested).__name__}:{cls._truncate_debug_snippet(nested, 80)}"
            )
        if expanded == 0:
          skipped.append(
            f"{idx}=list:{cls._truncate_debug_snippet(item, 80)}"
          )
        continue
      skipped.append(
        f"{idx}={type(item).__name__}:{cls._truncate_debug_snippet(item, 80)}"
      )
    if skipped:
      logger.warning(
        "观众记忆 JSON 列表内存在非 dict 项，已跳过/浅展开: %s | raw=%s",
        "; ".join(skipped[:5]),
        cls._truncate_debug_snippet(raw_snippet),
      )
    return normalized

  def retrieve_active_only(self) -> tuple[str, str, str]:
    active_memories = self._active.get_all()
    if not active_memories:
      return "", "", ""
    lines = ["【近期记忆】"]
    for memory in active_memories:
      lines.append(f"- {memory.content}")
      if memory.response:
        lines.append(f"  → 我说：「{memory.response[:80]}」")
    return "\n".join(lines), "", ""

  def compile_structured_context(
    self,
    query: Union[str, list[str]] = "",
    viewer_ids: Optional[list[str]] = None,
    include_persona: bool = True,
    include_corpus: bool = False,
    include_external_knowledge: bool = False,
    recall_profile: str = "deep_recall",
  ) -> str:
    if self._structured_retriever is None:
      return ""
    return self._structured_retriever.compile_prompt_context(
      query=query,
      viewer_ids=viewer_ids,
      include_persona=include_persona,
      include_corpus=include_corpus,
      include_external_knowledge=include_external_knowledge,
      recall_profile=recall_profile,
    )

  def _refresh_user_structured_indexes(self, viewer_ids: set[str]) -> None:
    if self._structured_retriever is None:
      return
    for viewer_id in viewer_ids:
      normalized = str(viewer_id).strip()
      if normalized:
        self._structured_retriever.rebuild_user_record(normalized)

  def _refresh_self_structured_indexes(self, include_threads: bool = False) -> None:
    if self._structured_retriever is None:
      return
    self._structured_retriever.rebuild_self_said_indexes()
    if include_threads:
      self._structured_retriever.rebuild_self_thread_index()

  async def record_interaction(
    self,
    user_input: str,
    response: str,
  ) -> None:
    """异步记录一次交互，并写入 active 层。"""
    try:
      model = self._get_summary_model()
      prompt = INTERACTION_SUMMARY_PROMPT.format(
        input=user_input,
        response=response,
      )
      summary = await model.ainvoke(prompt)
      summary_text = summary.content if hasattr(summary, "content") else str(summary)
      summary_text = summary_text.strip()
      if summary_text:
        self._active.add(summary_text, response=response)
        self._recent_interactions.append((user_input, response, datetime.now()))
        logger.debug("记录交互记忆: %s", summary_text)
    except Exception as e:
      logger.error("记录交互记忆失败: %s", e)

  def record_interaction_sync(
    self,
    user_input: str,
    response: str,
  ) -> None:
    summary_text = f"我回复了一位观众：他说「{user_input}」，我说了「{response[:50]}」"
    self._active.add(summary_text, response=response)
    self._recent_interactions.append((user_input, response, datetime.now()))

  async def record_viewer_memories(
    self,
    comments: list,
    ai_response_summary: str = "",
  ) -> None:
    """从观众弹幕中提取结构化用户记忆。"""
    if self._user_memory_store is None:
      return

    candidates = []
    for comment in comments:
      content = getattr(comment, "content", "").strip()
      if len(content) <= 2:
        continue
      if self._NOISE_DANMAKU.match(content):
        continue
      user_id = getattr(comment, "user_id", "")
      nickname = getattr(comment, "nickname", "未知")
      if not user_id:
        continue
      candidates.append({
        "user_id": user_id,
        "nickname": nickname,
        "content": content,
      })

    if not candidates:
      return

    comments_text = "\n".join(
      f"{idx}. {item['nickname']}：{item['content']}"
      for idx, item in enumerate(candidates)
    )
    try:
      prompt = VIEWER_SUMMARY_PROMPT.format(
        comments=comments_text,
        ai_response=ai_response_summary[:200] if ai_response_summary else "（无）",
      )
      model = self._get_summary_model()
      result = await model.ainvoke(prompt)
      text = result.content if hasattr(result, "content") else str(result)
      text = text.strip()

      if text.startswith(("[", "{")):
        raw_json = text
      else:
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        object_match = re.search(r"\{.*\}", text, re.DOTALL) if not json_match else None
        if json_match:
          raw_json = json_match.group()
        elif object_match:
          raw_json = object_match.group()
        else:
          logger.debug("观众记忆 LLM 返回无有效 JSON: %s", text[:100])
          return
      memories = json_repair.loads(raw_json)
      if not isinstance(memories, list):
        logger.debug(
          "观众记忆 JSON 解析结果非 list: %s | raw=%s",
          type(memories).__name__,
          self._truncate_debug_snippet(raw_json),
        )
        return
      memory_items = self._normalize_viewer_memory_items(memories, raw_json)
      if not memory_items:
        logger.debug(
          "观众记忆 JSON 没有可用 dict 项: raw=%s",
          self._truncate_debug_snippet(raw_json),
        )
        return

      touched_viewers: set[str] = set()
      for item in memory_items:
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
          continue
        src = candidates[idx]
        identity = item.get("identity")
        stable_facts = item.get("stable_facts")
        recent_state = item.get("recent_state")
        topic_profile = item.get("topic_profile")
        callbacks = item.get("callbacks")
        open_threads = item.get("open_threads")
        sensitive_topics = item.get("sensitive_topics")
        relationship_state = item.get("relationship_state")

        if not stable_facts and not callbacks and not recent_state and not open_threads:
          memory_text = str(item.get("memory", "")).strip()
          if memory_text:
            stable_facts = [{
              "fact": memory_text,
              "confidence": 0.60,
              "ttl_days": 30,
              "source": "viewer_summary_extract",
            }]

        normalized_facts = [
          fact for fact in (stable_facts or [])
          if isinstance(fact, dict) and str(fact.get("fact", "")).strip()
        ]
        normalized_recent_state = [
          state for state in (recent_state or [])
          if isinstance(state, dict) and str(state.get("fact", "")).strip()
        ]
        normalized_topic_profile = [
          topic for topic in (topic_profile or [])
          if isinstance(topic, dict) and str(topic.get("topic", "")).strip()
        ]
        normalized_callbacks = [
          hook for hook in (callbacks or [])
          if isinstance(hook, dict) and str(hook.get("hook", "")).strip()
        ]
        normalized_open_threads = [
          thread for thread in (open_threads or [])
          if isinstance(thread, dict) and str(thread.get("thread", "")).strip()
        ]
        normalized_sensitive_topics = [
          topic for topic in (sensitive_topics or [])
          if isinstance(topic, dict) and str(topic.get("topic", "")).strip()
        ]
        normalized_identity = identity if isinstance(identity, dict) else None
        normalized_relationship_state = relationship_state if isinstance(relationship_state, dict) else None
        source_mentions_guard_topic = self._mentions_guard_topic(src["content"])

        normalized_facts = self._sanitize_guard_fact_entries(normalized_facts, "fact")
        normalized_recent_state = self._sanitize_guard_fact_entries(normalized_recent_state, "fact")
        normalized_topic_profile = self._sanitize_guard_topic_entries(
          normalized_topic_profile,
          source_mentions_guard_topic=source_mentions_guard_topic,
        )
        normalized_callbacks = self._sanitize_guard_callback_entries(normalized_callbacks)
        normalized_open_threads = self._sanitize_guard_thread_entries(
          normalized_open_threads,
          source_mentions_guard_topic=source_mentions_guard_topic,
        )
        normalized_relationship_state = self._sanitize_guard_relationship_state(
          normalized_relationship_state,
          source_mentions_guard_topic=source_mentions_guard_topic,
        )

        has_meaningful_update = any((
          normalized_identity,
          normalized_facts,
          normalized_recent_state,
          normalized_topic_profile,
          normalized_relationship_state,
          normalized_callbacks,
          normalized_open_threads,
          normalized_sensitive_topics,
        ))
        if not has_meaningful_update:
          continue

        self._user_memory_store.record_extract(
          viewer_id=src["user_id"],
          nickname=src["nickname"],
          identity=normalized_identity,
          stable_facts=normalized_facts,
          recent_state=normalized_recent_state,
          topic_profile=normalized_topic_profile,
          relationship_state=normalized_relationship_state,
          callbacks=normalized_callbacks,
          open_threads=normalized_open_threads,
          sensitive_topics=normalized_sensitive_topics,
          legacy_source="viewer_summary_extract",
          was_addressed=bool(ai_response_summary),
        )
        touched_viewers.add(src["user_id"])

      self._refresh_user_structured_indexes(touched_viewers)
    except Exception as e:
      logger.error(
        "观众记忆提取失败: %s | raw=%s",
        e,
        self._truncate_debug_snippet(text if "text" in locals() else ""),
      )

  async def extract_stances(
    self,
    response: str,
    context: str = "",
  ) -> None:
    """从 AI 回复中提取立场/观点并写入 structured self memory。"""
    if self._self_memory_store is None:
      return
    if not self._may_contain_stance(response):
      return
    try:
      await self._extract_and_store_stances(context, response)
    except Exception as e:
      logger.error("立场提取失败: %s", e)

  def _may_contain_stance(self, response: str) -> bool:
    return bool(self._STANCE_INDICATORS.search(response))

  async def _extract_and_store_stances(
    self,
    user_input: str,
    response: str,
  ) -> None:
    if self._self_memory_store is None:
      return

    model = self._get_summary_model()
    prompt = STANCE_EXTRACTION_PROMPT.format(
      input=user_input,
      response=response,
    )
    result = await model.ainvoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
      return

    try:
      data = json.loads(json_match.group())
    except json.JSONDecodeError:
      logger.debug("立场提取 JSON 解析失败: %s", text[:100])
      return

    has_stance = bool(data.get("has_stance")) or bool(data.get("has_memory"))
    if not has_stance:
      return

    stances = data.get("stances", [])
    self_said = data.get("self_said", [])
    commitments = data.get("commitments", [])
    updated_self_memory = False

    if not self_said and stances:
      self_said = [
        {
          "topic": item.get("topic", ""),
          "statement": item.get("stance", ""),
        }
        for item in stances
        if isinstance(item, dict)
      ]

    for item in self_said:
      if not isinstance(item, dict):
        continue
      topic = str(item.get("topic", "")).strip()
      statement_text = str(
        item.get("statement", "") or item.get("stance", "")
      ).strip()
      if not statement_text:
        continue
      self._self_memory_store.record_stance(
        topic=topic,
        statement=statement_text,
        response_excerpt=response[:200],
        source="stance_extraction",
      )
      updated_self_memory = True

    for item in commitments:
      if not isinstance(item, dict):
        continue
      text_value = str(item.get("text", "")).strip()
      if not text_value:
        continue
      topic = str(item.get("topic", "")).strip()
      status = str(item.get("status", "open")).strip() or "open"
      self._self_memory_store.add_commitment(
        text=text_value,
        topic=topic,
        source="stance_extraction",
        status=status,
      )
      updated_self_memory = True

    if updated_self_memory:
      self._refresh_self_structured_indexes(include_threads=False)

  async def start(self) -> None:
    if self._summary_task is None:
      self._summary_task = asyncio.create_task(self._summary_loop())
      logger.info("记忆定时汇总任务已启动")

  async def stop(self) -> None:
    if self._summary_task is not None:
      self._summary_task.cancel()
      try:
        await self._summary_task
      except asyncio.CancelledError:
        pass
      logger.info("记忆定时汇总任务已停止")
    self._summary_task = None

  def clear_runtime_state(self) -> None:
    self._active.clear()
    self._recent_interactions.clear()
    logger.info("已清空运行期记忆状态")

  async def _summary_loop(self) -> None:
    interval = self._config.summary.interval_seconds
    while True:
      try:
        await asyncio.sleep(interval)
        await self._do_summary()
      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error("定时汇总出错: %s", e)

  async def _do_summary(self) -> None:
    if self._self_memory_store is None:
      return

    active_memories = self._active.get_all()
    active_texts = [memory.content for memory in active_memories]
    recent = list(self._recent_interactions)
    if not active_texts and not recent:
      return

    active_str = "\n".join(f"- {text}" for text in active_texts) if active_texts else "（无）"
    recent_str = (
      "\n".join(
        f"- 观众说「{inp}」，我回复了「{resp[:50]}」"
        for inp, resp, _ in recent
      )
      if recent else "（无）"
    )

    prompt = PERIODIC_SUMMARY_PROMPT.format(
      active_memories=active_str,
      recent_interactions=recent_str,
    )

    try:
      model = self._get_summary_model()
      result = await model.ainvoke(prompt)
      summary_text = result.content if hasattr(result, "content") else str(result)
      summary_text = summary_text.strip()
      if not summary_text:
        return

      self._self_memory_store.add_thread_memory(
        summary_text,
        source_layer="summary_rollup",
      )
      self._refresh_self_structured_indexes(include_threads=True)
      self._recent_interactions.clear()
      logger.info("定时汇总完成: %s", summary_text[:60])
    except Exception as e:
      logger.error("定时汇总 LLM 调用失败: %s", e)

  def debug_state(self) -> dict:
    active_memories = self._active.get_all()
    result = {
      "active_count": self._active.count(),
      "active_capacity": self._active._config.capacity,
      "active_memories": [
        {
          "content": memory.content,
          "timestamp": memory.timestamp.strftime("%H:%M:%S"),
          "response": memory.response,
        }
        for memory in active_memories
      ],
      "recent_interactions": len(self._recent_interactions),
      "summary_task_running": self._summary_task is not None and not self._summary_task.done(),
      "legacy_layers_removed": True,
    }

    if self._user_memory_store is not None:
      result["user_memory_store"] = self._user_memory_store.debug_state()
    if self._self_memory_store is not None:
      result["self_memory_store"] = self._self_memory_store.debug_state()
    if self._persona_spec_store is not None:
      result["persona_spec_store"] = self._persona_spec_store.debug_state()
    if self._corpus_store is not None:
      result["corpus_store"] = self._corpus_store.debug_state()
    if self._external_knowledge_store is not None:
      result["external_knowledge_store"] = self._external_knowledge_store.debug_state()
    if self._structured_retriever is not None:
      result["structured_projection_indexes"] = self._structured_retriever.debug_state()

    return result
