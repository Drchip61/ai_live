"""
structured 主检索器

以 JSON 结构化真相源为主库，并维护一组面向检索的 Chroma 投影索引。
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from .config import EmbeddingConfig, StructuredContextConfig
from .context_schema import (
  CompiledMemoryContext,
  CorpusEntry,
  ExternalKnowledgeEntry,
  PersonaSpecRecord,
  SelfMemoryRecord,
  UserMemoryRecord,
  resolve_preferred_address,
)
from .context_store import (
  CorpusStore,
  ExternalKnowledgeStore,
  PersonaSpecStore,
  SelfMemoryStore,
  UserMemoryStore,
)
from .store import VectorStore


def _stable_id(*parts: str) -> str:
  raw = "||".join(str(part or "").strip() for part in parts)
  return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _safe_float(value, default: float = 0.0) -> float:
  try:
    return float(value or default)
  except (TypeError, ValueError):
    return default


def _stability_score(text: str) -> float:
  normalized = str(text or "").strip()
  if not normalized:
    return 0.0
  score = 0.0
  stable_markers = (
    "经常", "习惯", "总是", "一直", "通常", "常常", "喜欢", "最喜欢",
    "追看", "关注", "白噪音", "入睡", "名字叫", "姓",
  )
  ephemeral_markers = (
    "当前", "今天", "明天", "今晚", "今早", "凌晨", "早上", "十分钟",
    "小时", "两天", "三天", "等待", "周报", "下班", "到家",
    "堵车", "外卖", "便当", "便利店", "当时", "周日晚上",
  )
  for marker in stable_markers:
    if marker in normalized:
      score += 1.0
  for marker in ephemeral_markers:
    if marker in normalized:
      score -= 1.0
  return score


@dataclass(frozen=True)
class RecallProfile:
  """结构化检索画像，决定 normal / deep_recall 的召回深度。"""

  max_viewers: int
  user_fact_top_k: int
  user_recent_state_top_k: int
  user_topic_top_k: int
  user_callback_top_k: int
  user_open_thread_top_k: int
  user_sensitive_top_k: int
  self_said_top_k: int
  self_commitment_top_k: int
  self_thread_top_k: int
  stable_preference_top_k: int


class StructuredMemoryRetriever:
  """基于 structured store 的主检索器"""

  def __init__(
    self,
    user_memory_store: Optional[UserMemoryStore],
    self_memory_store: Optional[SelfMemoryStore],
    persona_spec_store: Optional[PersonaSpecStore],
    corpus_store: Optional[CorpusStore],
    external_knowledge_store: Optional[ExternalKnowledgeStore],
    embedding_config: EmbeddingConfig,
    embeddings,
    config: StructuredContextConfig,
  ) -> None:
    self._config = config
    self._user_memory_store = user_memory_store
    self._self_memory_store = self_memory_store
    self._persona_spec_store = persona_spec_store
    self._corpus_store = corpus_store
    self._external_knowledge_store = external_knowledge_store

    prefix = config.collection_prefix
    self._user_fact_index = VectorStore(f"{prefix}user_facts", embedding_config, embeddings=embeddings)
    self._user_callback_index = VectorStore(f"{prefix}user_callbacks", embedding_config, embeddings=embeddings)
    self._self_said_index = VectorStore(f"{prefix}self_said", embedding_config, embeddings=embeddings)
    self._self_commitment_index = VectorStore(f"{prefix}self_commitments", embedding_config, embeddings=embeddings)
    self._self_thread_index = VectorStore(f"{prefix}self_threads", embedding_config, embeddings=embeddings)
    self._persona_index = VectorStore(f"{prefix}persona_spec", embedding_config, embeddings=embeddings)
    self._corpus_index = VectorStore(f"{prefix}corpus", embedding_config, embeddings=embeddings)
    self._knowledge_index = VectorStore(f"{prefix}external_knowledge", embedding_config, embeddings=embeddings)
    self._last_retrieval_trace: dict[str, Any] = {}

  def rebuild_all(self) -> None:
    self.rebuild_user_indexes()
    self.rebuild_self_said_indexes()
    self.rebuild_self_thread_index()
    self.rebuild_persona_index()
    self.rebuild_corpus_index()
    self.rebuild_knowledge_index()

  def rebuild_user_indexes(self) -> None:
    fact_ids: list[str] = []
    fact_contents: list[str] = []
    fact_metadatas: list[dict] = []
    callback_ids: list[str] = []
    callback_contents: list[str] = []
    callback_metadatas: list[dict] = []
    if self._user_memory_store is not None:
      for record in self._user_memory_store.all().values():
        fact_docs, callback_docs = self._build_user_record_docs(record)
        fact_ids.extend(fact_docs[0])
        fact_contents.extend(fact_docs[1])
        fact_metadatas.extend(fact_docs[2])
        callback_ids.extend(callback_docs[0])
        callback_contents.extend(callback_docs[1])
        callback_metadatas.extend(callback_docs[2])
    self._user_fact_index.replace_all(fact_ids, fact_contents, fact_metadatas)
    self._user_callback_index.replace_all(callback_ids, callback_contents, callback_metadatas)

  def rebuild_user_record(self, viewer_id: str) -> None:
    fact_ids: list[str] = []
    fact_contents: list[str] = []
    fact_metadatas: list[dict] = []
    callback_ids: list[str] = []
    callback_contents: list[str] = []
    callback_metadatas: list[dict] = []
    if self._user_memory_store is not None:
      record = self._user_memory_store.get(viewer_id)
      if record is not None:
        fact_docs, callback_docs = self._build_user_record_docs(record)
        fact_ids, fact_contents, fact_metadatas = fact_docs
        callback_ids, callback_contents, callback_metadatas = callback_docs
    self._user_fact_index.replace_where(
      {"viewer_id": viewer_id},
      fact_ids,
      fact_contents,
      fact_metadatas,
    )
    self._user_callback_index.replace_where(
      {"viewer_id": viewer_id},
      callback_ids,
      callback_contents,
      callback_metadatas,
    )

  def rebuild_self_said_indexes(self) -> None:
    said_ids: list[str] = []
    said_contents: list[str] = []
    said_metadatas: list[dict] = []
    commitment_ids: list[str] = []
    commitment_contents: list[str] = []
    commitment_metadatas: list[dict] = []
    if self._self_memory_store is not None:
      self_record = self._self_memory_store.get()
      said_ids, said_contents, said_metadatas = self._build_self_said_docs(self_record)
      commitment_ids, commitment_contents, commitment_metadatas = self._build_commitment_docs(self_record)
    self._self_said_index.replace_all(said_ids, said_contents, said_metadatas)
    self._self_commitment_index.replace_all(
      commitment_ids,
      commitment_contents,
      commitment_metadatas,
    )

  def rebuild_self_thread_index(self) -> None:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    if self._self_memory_store is not None:
      self_record = self._self_memory_store.get()
      doc_ids, contents, metadatas = self._build_self_thread_docs(self_record)
    self._self_thread_index.replace_all(doc_ids, contents, metadatas)

  def rebuild_persona_index(self) -> None:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    if self._persona_spec_store is not None:
      record = self._persona_spec_store.get()
      for item in record.items:
        text = str(item.get("text", "")).strip()
        if not text:
          continue
        section = str(item.get("section", "")).strip()
        doc_ids.append(_stable_id("persona", section, text))
        contents.append(f"{section}：{text}" if section else text)
        metadatas.append({
          "section": section,
          "display_text": f"{section}：{text}" if section else text,
        })
    self._persona_index.replace_all(doc_ids, contents, metadatas)

  def rebuild_corpus_index(self) -> None:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    if self._corpus_store is not None:
      entries = self._corpus_store.list_enabled()
      for entry in entries:
        tags = " ".join(entry.style_tags + entry.scene_tags)
        contents.append(f"{entry.kind} {tags} {entry.text}".strip())
        doc_ids.append(_stable_id("corpus", entry.corpus_id, entry.text))
        metadatas.append({
          "kind": entry.kind,
          "display_text": entry.text,
          "style_tags": ",".join(entry.style_tags),
          "scene_tags": ",".join(entry.scene_tags),
          "quality_score": entry.quality_score,
        })
    self._corpus_index.replace_all(doc_ids, contents, metadatas)

  def rebuild_knowledge_index(self) -> None:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    if self._external_knowledge_store is not None:
      entries = self._external_knowledge_store.list_enabled()
      for entry in entries:
        head = entry.topic or entry.category
        stance = str(entry.streamer_stance or "").strip()
        content = f"{head} {entry.summary} {stance}".strip()
        if not content:
          continue
        display_text = f"{head}：{entry.summary}" if head and entry.summary else content
        if stance:
          display_text = f"{display_text}\n主播立场：{stance}"
        doc_ids.append(_stable_id("knowledge", entry.knowledge_id, content))
        contents.append(content)
        metadatas.append({
          "topic": entry.topic,
          "category": entry.category,
          "display_text": display_text,
          "reliability": entry.reliability,
        })
    self._knowledge_index.replace_all(doc_ids, contents, metadatas)

  def build_compiled_context(
    self,
    query: Union[str, list[str]],
    viewer_ids: Optional[list[str]] = None,
    include_persona: bool = True,
    include_corpus: bool = False,
    include_external_knowledge: bool = False,
    recall_profile: str = "deep_recall",
  ) -> CompiledMemoryContext:
    query_text = self._normalize_query(query)
    query_embed_ms = 0.0
    query_embedding = None
    if query_text:
      embed_started = time.monotonic()
      query_embedding = self._query_embedding(query_text)
      query_embed_ms = (time.monotonic() - embed_started) * 1000
    profile = self._resolve_recall_profile(recall_profile)
    normalized_viewers = [
      str(viewer_id).strip()
      for viewer_id in viewer_ids or []
      if str(viewer_id).strip()
    ]
    request_trace: dict[str, Any] = {
      "semantic_search_count": 0,
      "query_embed_count": 1 if query_embedding else 0,
      "query_embed_ms": round(query_embed_ms, 1),
      "viewer_count": len(list(dict.fromkeys(normalized_viewers))[:profile.max_viewers]),
      "recall_profile": recall_profile,
      "vector_searches": [],
    }
    context = CompiledMemoryContext(
      user_memory_lines=tuple(self._build_user_lines(query_text, viewer_ids, profile, query_embedding, trace=request_trace)),
      self_memory_lines=tuple(self._build_self_lines(query_text, profile, query_embedding, trace=request_trace)),
      persona_lines=tuple(self._build_persona_lines(query_text, query_embedding, trace=request_trace)) if include_persona else (),
      corpus_lines=tuple(self._build_corpus_lines(query_text, query_embedding, trace=request_trace)) if include_corpus else (),
      knowledge_lines=tuple(self._build_knowledge_lines(query_text, query_embedding, trace=request_trace)) if include_external_knowledge else (),
    )
    vector_searches = request_trace.get("vector_searches", [])
    request_trace["chroma_query_ms"] = round(sum(
      float(item.get("chroma_query_ms", 0.0) or 0.0)
      for item in vector_searches
    ), 1)
    request_trace["self_heal_ms"] = round(sum(
      float(item.get("self_heal_ms", 0.0) or 0.0)
      for item in vector_searches
    ), 1)
    request_trace["retry_count"] = int(sum(
      int(item.get("retry_count", 0) or 0)
      for item in vector_searches
    ))
    self._last_retrieval_trace = request_trace
    return context

  def compile_prompt_context(
    self,
    query: Union[str, list[str]],
    viewer_ids: Optional[list[str]] = None,
    include_persona: bool = True,
    include_corpus: bool = False,
    include_external_knowledge: bool = False,
    recall_profile: str = "deep_recall",
  ) -> str:
    context = self.build_compiled_context(
      query=query,
      viewer_ids=viewer_ids,
      include_persona=include_persona,
      include_corpus=include_corpus,
      include_external_knowledge=include_external_knowledge,
      recall_profile=recall_profile,
    )
    return context.to_prompt_blocks()

  def compile_prompt_context_with_trace(
    self,
    query: Union[str, list[str]],
    viewer_ids: Optional[list[str]] = None,
    include_persona: bool = True,
    include_corpus: bool = False,
    include_external_knowledge: bool = False,
    recall_profile: str = "deep_recall",
  ) -> tuple[str, dict[str, Any]]:
    prompt_context = self.compile_prompt_context(
      query=query,
      viewer_ids=viewer_ids,
      include_persona=include_persona,
      include_corpus=include_corpus,
      include_external_knowledge=include_external_knowledge,
      recall_profile=recall_profile,
    )
    return prompt_context, self.get_last_retrieval_trace()

  def debug_state(self) -> dict:
    return {
      "user_fact_docs": len(self._user_fact_index.get_all().get("ids", [])),
      "user_callback_docs": len(self._user_callback_index.get_all().get("ids", [])),
      "self_said_docs": len(self._self_said_index.get_all().get("ids", [])),
      "self_commitment_docs": len(self._self_commitment_index.get_all().get("ids", [])),
      "self_thread_docs": len(self._self_thread_index.get_all().get("ids", [])),
      "persona_docs": len(self._persona_index.get_all().get("ids", [])),
      "corpus_docs": len(self._corpus_index.get_all().get("ids", [])),
      "knowledge_docs": len(self._knowledge_index.get_all().get("ids", [])),
    }

  def get_last_retrieval_trace(self) -> dict[str, Any]:
    vector_searches = [
      dict(item)
      for item in self._last_retrieval_trace.get("vector_searches", [])
      if isinstance(item, dict)
    ]
    return {
      key: value
      for key, value in {
        **self._last_retrieval_trace,
        "vector_searches": vector_searches,
      }.items()
    }

  def ensure_healthy(self) -> None:
    for index in (
      self._user_fact_index,
      self._user_callback_index,
      self._self_said_index,
      self._self_commitment_index,
      self._self_thread_index,
      self._persona_index,
      self._corpus_index,
      self._knowledge_index,
    ):
      index.ensure_healthy()

  def _build_user_record_docs(
    self,
    record: UserMemoryRecord,
  ) -> tuple[
    tuple[list[str], list[str], list[dict]],
    tuple[list[str], list[str], list[dict]],
  ]:
    identity = record.identity or {}
    nicknames = tuple(identity.get("nicknames", ()))
    nickname = resolve_preferred_address(
      identity,
      fallback_nicknames=nicknames,
      raw_aliases=(record.viewer_id,),
      fallback=record.viewer_id,
    )

    fact_ids: list[str] = []
    fact_contents: list[str] = []
    fact_metadatas: list[dict] = []
    for item in record.stable_facts:
      fact = str(item.get("fact", "")).strip()
      if not fact:
        continue
      fact_ids.append(_stable_id("user_fact", record.viewer_id, fact))
      fact_contents.append(fact)
      fact_metadatas.append({
        "viewer_id": record.viewer_id,
        "nickname": nickname,
        "display_text": fact,
        "confidence": _safe_float(item.get("confidence"), 0.0),
        "freshness": _safe_float(item.get("freshness"), 0.0),
        "stability_score": _stability_score(fact),
      })

    callback_ids: list[str] = []
    callback_contents: list[str] = []
    callback_metadatas: list[dict] = []
    for item in record.callbacks:
      hook = str(item.get("hook", "")).strip()
      if not hook:
        continue
      callback_ids.append(_stable_id("user_callback", record.viewer_id, hook))
      callback_contents.append(hook)
      callback_metadatas.append({
        "viewer_id": record.viewer_id,
        "nickname": nickname,
        "display_text": hook,
        "confidence": _safe_float(item.get("confidence"), 0.0),
        "freshness": _safe_float(item.get("freshness"), 0.0),
        "stability_score": _stability_score(hook),
      })

    return (
      (fact_ids, fact_contents, fact_metadatas),
      (callback_ids, callback_contents, callback_metadatas),
    )

  def _build_self_said_docs(
    self,
    self_record: SelfMemoryRecord,
  ) -> tuple[list[str], list[str], list[dict]]:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    for item in self_record.self_said:
      text = str(item.get("text", "")).strip()
      if not text:
        continue
      topic = str(item.get("topic", "")).strip()
      doc_ids.append(_stable_id("self_said", topic, text))
      contents.append(f"{topic} {text}".strip())
      metadatas.append({
        "topic": topic,
        "display_text": text,
        "confidence": _safe_float(item.get("confidence"), 0.0),
      })
    return doc_ids, contents, metadatas

  def _build_commitment_docs(
    self,
    self_record: SelfMemoryRecord,
  ) -> tuple[list[str], list[str], list[dict]]:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    for item in self_record.commitments:
      text = str(item.get("text", "")).strip()
      if not text:
        continue
      topic = str(item.get("topic", "")).strip()
      status = str(item.get("status", "")).strip()
      doc_ids.append(_stable_id("self_commitment", topic, text))
      contents.append(f"{topic} {text}".strip())
      metadatas.append({
        "topic": topic,
        "status": status,
        "display_text": text,
      })
    return doc_ids, contents, metadatas

  def _build_self_thread_docs(
    self,
    self_record: SelfMemoryRecord,
  ) -> tuple[list[str], list[str], list[dict]]:
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    for item in self_record.self_threads:
      text = str(item.get("text", "")).strip()
      if not text:
        continue
      source_layer = str(item.get("source_layer", "")).strip()
      doc_ids.append(_stable_id("self_thread", source_layer, text))
      contents.append(text)
      metadatas.append({
        "source_layer": source_layer,
        "display_text": text,
      })
    return doc_ids, contents, metadatas

  @staticmethod
  def _normalize_query(query: Union[str, list[str]]) -> str:
    if isinstance(query, list):
      parts = [str(item).strip() for item in query if str(item).strip()]
      return " ".join(parts)
    return str(query or "").strip()

  def _query_embedding(self, query_text: str) -> Optional[list[float]]:
    normalized = str(query_text or "").strip()
    if not normalized:
      return None
    return self._user_fact_index.embed_query(normalized)

  def _resolve_recall_profile(self, recall_profile: str) -> RecallProfile:
    if recall_profile == "normal":
      return RecallProfile(
        max_viewers=1 if self._config.max_viewers > 0 else 0,
        user_fact_top_k=min(2, self._config.user_fact_top_k),
        user_recent_state_top_k=min(1, self._config.user_recent_state_top_k),
        user_topic_top_k=min(2, self._config.user_topic_top_k),
        user_callback_top_k=min(1, self._config.user_callback_top_k),
        user_open_thread_top_k=min(1, self._config.user_open_thread_top_k),
        user_sensitive_top_k=min(1, self._config.user_sensitive_top_k),
        self_said_top_k=min(1, self._config.self_said_top_k),
        self_commitment_top_k=min(1, self._config.self_commitment_top_k),
        self_thread_top_k=min(1, self._config.self_thread_top_k),
        stable_preference_top_k=1,
      )
    return RecallProfile(
      max_viewers=self._config.max_viewers,
      user_fact_top_k=self._config.user_fact_top_k,
      user_recent_state_top_k=self._config.user_recent_state_top_k,
      user_topic_top_k=self._config.user_topic_top_k,
      user_callback_top_k=self._config.user_callback_top_k,
      user_open_thread_top_k=self._config.user_open_thread_top_k,
      user_sensitive_top_k=self._config.user_sensitive_top_k,
      self_said_top_k=self._config.self_said_top_k,
      self_commitment_top_k=self._config.self_commitment_top_k,
      self_thread_top_k=self._config.self_thread_top_k,
      stable_preference_top_k=2,
    )

  def _build_user_lines(
    self,
    query_text: str,
    viewer_ids: Optional[list[str]],
    profile: RecallProfile,
    query_embedding: Optional[list[float]] = None,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if self._user_memory_store is None:
      return []
    picked_viewers: list[str] = []
    for viewer_id in viewer_ids or []:
      normalized = str(viewer_id).strip()
      if normalized and normalized not in picked_viewers:
        picked_viewers.append(normalized)
      if len(picked_viewers) >= profile.max_viewers:
        break
    if not picked_viewers:
      return []

    lines: list[str] = []
    for viewer_id in picked_viewers:
      record = self._user_memory_store.get(viewer_id)
      if record is None:
        continue
      if not lines:
        lines.append("使用原则：最多只轻轻打一张关系牌，不要背档案式复述历史。")
      identity = record.identity or {}
      nicknames = tuple(identity.get("nicknames", ()))
      names = tuple(identity.get("names", ()))
      preferred_address = resolve_preferred_address(
        identity,
        fallback_nicknames=nicknames,
        raw_aliases=(record.viewer_id,),
        fallback=record.viewer_id,
      )
      nickname = preferred_address
      lines.append(f"当前关注对象：{nickname}")

      sensitive_topics = self._pick_sensitive_entries(
        record.sensitive_topics,
        limit=profile.user_sensitive_top_k,
      )
      if sensitive_topics:
        lines.append(f"{nickname} 的边界提醒：" + "；".join(sensitive_topics))

      identity_parts: list[str] = []
      if preferred_address:
        identity_parts.append(f"建议称呼={preferred_address}")
      clean_nicknames = self._clean_alias_list(nicknames)
      clean_names = self._clean_alias_list(names)
      all_aliases = list(dict.fromkeys(
        alias for alias in (clean_nicknames + clean_names)
        if alias != preferred_address and alias != record.viewer_id
      ))
      if all_aliases:
        identity_parts.append("曾用名/别名=" + "/".join(all_aliases[:5]))
      elif clean_names:
        identity_parts.append("名字线索=" + "/".join(clean_names[:2]))
      occupation = identity.get("occupation", {}) or {}
      occupation_value = str(occupation.get("value", "")).strip() if isinstance(occupation, dict) else ""
      if occupation_value:
        identity_parts.append(f"职业={occupation_value}")
      if identity_parts:
        lines.append(f"{nickname} 的身份信息：" + "，".join(identity_parts))

      state_parts: list[str] = []
      state = record.relationship_state or {}
      familiarity = state.get("familiarity")
      trust = state.get("trust")
      tease_threshold = state.get("tease_threshold")
      interaction_style = state.get("interaction_style")
      address_style = state.get("address_style")
      public_ack_count = state.get("public_ack_count")
      publicly_acknowledged = state.get("publicly_acknowledged")
      last_dialogue_stop = str(state.get("last_dialogue_stop", "")).strip()
      if familiarity not in (None, ""):
        state_parts.append(f"熟悉度={familiarity}")
      if trust not in (None, ""):
        state_parts.append(f"信任度={trust}")
      if tease_threshold not in (None, ""):
        state_parts.append(f"玩笑阈值={tease_threshold}")
      if interaction_style:
        state_parts.append(f"互动风格={interaction_style}")
      if address_style:
        state_parts.append(f"称呼方式={address_style}")
      if public_ack_count not in (None, ""):
        state_parts.append(f"被公开接住次数={public_ack_count}")
      elif publicly_acknowledged:
        state_parts.append("被公开接住过=是")
      if last_dialogue_stop:
        state_parts.append(f"上次停在={last_dialogue_stop}")
      if state_parts:
        lines.append(f"{nickname} 的关系状态：" + "，".join(state_parts))

      facts = self._search_user_memories(
        self._user_fact_index,
        record,
        query_text=query_text,
        query_embedding=query_embedding,
        top_k=profile.user_fact_top_k,
        fallback_items=record.stable_facts,
        text_key="fact",
        trace=trace,
      )
      if facts:
        lines.append(f"{nickname} 的稳定事实：" + "；".join(facts))

      recent_state = self._fallback_texts(
        record.recent_state,
        "fact",
        profile.user_recent_state_top_k,
        prefer_recent=True,
      )
      if recent_state:
        lines.append(f"{nickname} 最近在忙/最近状态：" + "；".join(recent_state))

      topic_lines = self._pick_topic_entries(record.topic_profile, profile.user_topic_top_k)
      if topic_lines:
        lines.append(f"{nickname} 常聊话题：" + "，".join(topic_lines))

      callbacks = self._search_user_memories(
        self._user_callback_index,
        record,
        query_text=query_text,
        query_embedding=query_embedding,
        top_k=profile.user_callback_top_k,
        fallback_items=record.callbacks,
        text_key="hook",
        trace=trace,
      )
      if callbacks:
        lines.append(f"{nickname} 的历史梗/回钩线索：" + "；".join(callbacks))

      open_threads = self._fallback_texts(
        record.open_threads,
        "thread",
        profile.user_open_thread_top_k,
        prefer_recent=True,
      )
      if open_threads:
        lines.append(f"{nickname} 上次对话停在：" + "；".join(open_threads))
    return lines

  def _build_self_lines(
    self,
    query_text: str,
    profile: RecallProfile,
    query_embedding: Optional[list[float]] = None,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if self._self_memory_store is None:
      return []
    record = self._self_memory_store.get()
    lines: list[str] = []

    self_said = self._search_or_fallback(
      self._self_said_index,
      query_text=query_text,
      query_embedding=query_embedding,
      top_k=profile.self_said_top_k,
      fallback_items=record.self_said,
      text_key="text",
      trace=trace,
    )
    if self_said:
      lines.append("和当前话题相关的我说过：" + "；".join(self_said))

    commitments = self._search_or_fallback(
      self._self_commitment_index,
      query_text=query_text,
      query_embedding=query_embedding,
      top_k=profile.self_commitment_top_k,
      fallback_items=record.commitments,
      text_key="text",
      trace=trace,
    )
    if commitments:
      lines.append("仍在延续的承诺/话头：" + "；".join(commitments))

    threads = self._search_or_fallback(
      self._self_thread_index,
      query_text=query_text,
      query_embedding=query_embedding,
      top_k=profile.self_thread_top_k,
      fallback_items=record.self_threads,
      text_key="text",
      trace=trace,
    )
    if threads:
      lines.append("可续接的旧线头：" + "；".join(threads))

    preferences = self._sort_items(record.stable_preferences, "text")[:profile.stable_preference_top_k]
    preference_texts = [str(item.get("text", "")).strip() for item in preferences if str(item.get("text", "")).strip()]
    if preference_texts:
      lines.append("较稳定的表达偏好：" + "；".join(preference_texts))
    return lines

  def _build_persona_lines(
    self,
    query_text: str,
    query_embedding: Optional[list[float]] = None,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if self._persona_spec_store is None:
      return []
    record = self._persona_spec_store.get()
    if query_text:
      lines = self._search_texts(
        self._persona_index,
        query_text,
        top_k=self._config.persona_top_k,
        query_embedding=query_embedding,
        trace=trace,
      )
      if lines:
        return lines
    result: list[str] = []
    for item in record.items[:self._config.persona_top_k]:
      section = str(item.get("section", "")).strip()
      text = str(item.get("text", "")).strip()
      if not text:
        continue
      result.append(f"{section}：{text}" if section else text)
    return result

  def _build_corpus_lines(
    self,
    query_text: str,
    query_embedding: Optional[list[float]] = None,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if self._corpus_store is None:
      return []
    if query_text:
      lines = self._search_texts(
        self._corpus_index,
        query_text,
        top_k=self._config.corpus_top_k,
        query_embedding=query_embedding,
        trace=trace,
      )
      if lines:
        return lines
    return [entry.text for entry in self._corpus_store.list_enabled()[:self._config.corpus_top_k]]

  @staticmethod
  def _split_corpus_tags(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
      return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())

  def retrieve_corpus_lines(
    self,
    query: Union[str, list[str]] = "",
    style_tag: str = "",
    scene_tag: str = "",
    top_k: Optional[int] = None,
  ) -> list[str]:
    if self._corpus_store is None:
      return []

    limit = max(1, int(top_k or self._config.corpus_top_k or 1))
    query_text = self._normalize_query(query)
    search_query = query_text or " ".join(
      part for part in (style_tag, scene_tag) if str(part or "").strip()
    ).strip() or "语料参考"
    candidate_k = max(limit * 4, limit)
    search_query_embedding = self._query_embedding(search_query)

    ranked: list[tuple[int, int, float, float, str]] = []
    seen: set[str] = set()
    for doc, score in self._corpus_index.search_by_vector(search_query_embedding or [], top_k=candidate_k):
      if self._score_too_far(score):
        continue
      meta = doc.metadata or {}
      text = str(meta.get("display_text", "")).strip() or str(doc.page_content or "").strip()
      if not text or text in seen:
        continue
      style_tags = self._split_corpus_tags(meta.get("style_tags", ""))
      scene_tags = self._split_corpus_tags(meta.get("scene_tags", ""))
      style_hit = int(bool(style_tag) and style_tag in style_tags)
      scene_hit = int(bool(scene_tag) and scene_tag in scene_tags)
      if style_tag and scene_tag:
        if not (style_hit or scene_hit):
          continue
      elif style_tag and not style_hit:
        continue
      elif scene_tag and not scene_hit:
        continue
      seen.add(text)
      ranked.append((
        style_hit + scene_hit,
        1 if (style_hit and scene_hit) else 0,
        _safe_float(meta.get("quality_score"), default=0.5),
        float(score or 0.0),
        text,
      ))

    if len(ranked) < limit:
      fallback_entries: list[CorpusEntry] = []
      if style_tag or scene_tag:
        fallback_entries.extend(
          self._corpus_store.get_by_tags(
            style_tag=style_tag,
            scene_tag=scene_tag,
            limit=limit,
          )
        )
        if not fallback_entries and style_tag and scene_tag:
          fallback_entries.extend(
            self._corpus_store.get_by_tags(style_tag=style_tag, limit=limit)
          )
          fallback_entries.extend(
            self._corpus_store.get_by_tags(scene_tag=scene_tag, limit=limit)
          )
      elif not query_text:
        fallback_entries.extend(self._corpus_store.list_enabled()[:limit])

      for entry in fallback_entries:
        text = str(entry.text or "").strip()
        if not text or text in seen:
          continue
        style_hit = int(bool(style_tag) and style_tag in entry.style_tags)
        scene_hit = int(bool(scene_tag) and scene_tag in entry.scene_tags)
        seen.add(text)
        ranked.append((
          style_hit + scene_hit,
          1 if (style_hit and scene_hit) else 0,
          float(entry.quality_score or 0.5),
          0.0,
          text,
        ))

    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    return [text for *_ignored, text in ranked[:limit]]

  def retrieve_corpus_context(
    self,
    query: Union[str, list[str]] = "",
    style_tag: str = "",
    scene_tag: str = "",
    top_k: Optional[int] = None,
  ) -> str:
    lines = self.retrieve_corpus_lines(
      query=query,
      style_tag=style_tag,
      scene_tag=scene_tag,
      top_k=top_k,
    )
    if not lines:
      return ""
    hints = []
    if style_tag:
      hints.append(f"风格={style_tag}")
    if scene_tag:
      hints.append(f"场景={scene_tag}")
    guidance = "借鉴以下语料的表达方式、节奏或梗感，用你自己的语气自然表达，不要直接照抄。"
    if hints:
      guidance += f"（{' | '.join(hints)}）"
    return guidance + "\n" + "\n".join(
      f"{idx}. {line}" for idx, line in enumerate(lines, 1)
    )

  def _build_knowledge_lines(
    self,
    query_text: str,
    query_embedding: Optional[list[float]] = None,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if self._external_knowledge_store is None:
      return []
    if query_text:
      lines = self._search_texts(
        self._knowledge_index,
        query_text,
        top_k=self._config.knowledge_top_k,
        query_embedding=query_embedding,
        trace=trace,
      )
      if lines:
        return lines
    result: list[str] = []
    for entry in self._external_knowledge_store.list_enabled()[:self._config.knowledge_top_k]:
      head = entry.topic or entry.category
      stance = str(entry.streamer_stance or "").strip()
      if entry.summary:
        line = f"{head}：{entry.summary}" if head else entry.summary
        if stance:
          line = f"{line}\n主播立场：{stance}"
        result.append(line)
    return result

  def _search_user_memories(
    self,
    index: VectorStore,
    record: UserMemoryRecord,
    query_text: str,
    query_embedding: Optional[list[float]],
    top_k: int,
    fallback_items: tuple[dict, ...],
    text_key: str,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if top_k <= 0:
      return []
    if query_text:
      lines = self._search_texts(
        index,
        query_text,
        top_k=top_k,
        where={"viewer_id": record.viewer_id},
        query_embedding=query_embedding,
        trace=trace,
      )
      if lines:
        return lines
    return self._fallback_texts(fallback_items, text_key, top_k)

  def _search_or_fallback(
    self,
    index: VectorStore,
    query_text: str,
    query_embedding: Optional[list[float]],
    top_k: int,
    fallback_items: tuple[dict, ...],
    text_key: str,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if top_k <= 0:
      return []
    if query_text:
      lines = self._search_texts(
        index,
        query_text,
        top_k=top_k,
        query_embedding=query_embedding,
        trace=trace,
      )
      if lines:
        return lines
    return self._fallback_texts(fallback_items, text_key, top_k)

  def _search_texts(
    self,
    index: VectorStore,
    query_text: str,
    top_k: int,
    where: Optional[dict] = None,
    query_embedding: Optional[list[float]] = None,
    *,
    trace: Optional[dict[str, Any]] = None,
  ) -> list[str]:
    if not query_text or top_k <= 0:
      return []
    trace_collector = None
    if trace is not None:
      trace["semantic_search_count"] = int(trace.get("semantic_search_count", 0) or 0) + 1
      trace_collector = trace.setdefault("vector_searches", [])
    picked: list[str] = []
    seen: set[str] = set()
    search_results = (
      index.search_by_vector(
        query_embedding,
        top_k=top_k,
        where=where,
        trace_collector=trace_collector,
      )
      if query_embedding is not None
      else index.search(
        query_text,
        top_k=top_k,
        where=where,
        trace_collector=trace_collector,
      )
    )
    for doc, score in search_results:
      if self._score_too_far(score):
        continue
      meta = doc.metadata or {}
      text = str(meta.get("display_text", "")).strip() or str(doc.page_content or "").strip()
      if not text or text in seen:
        continue
      seen.add(text)
      picked.append(text)
      if len(picked) >= top_k:
        break
    return picked

  def _score_too_far(self, score: Optional[float]) -> bool:
    threshold = getattr(self._config, "semantic_max_distance", None)
    if score is None or threshold in (None, ""):
      return False
    try:
      return float(score) > float(threshold)
    except (TypeError, ValueError):
      return False

  def _fallback_texts(
    self,
    items: tuple[dict, ...],
    text_key: str,
    limit: int,
    prefer_recent: bool = False,
  ) -> list[str]:
    result: list[str] = []
    for item in self._sort_items(items, text_key, prefer_recent=prefer_recent)[:limit]:
      text = str(item.get(text_key, "")).strip()
      if text:
        result.append(text)
    return result

  @staticmethod
  def _pick_direct_entries(
    items: tuple[dict, ...],
    key_name: str,
    value_name: str,
    limit: int,
  ) -> list[str]:
    result: list[str] = []
    sorted_items = sorted(
      items,
      key=lambda item: (
        _safe_float(item.get("confidence"), 0.0),
        _safe_float(item.get("freshness"), 0.0),
        str(item.get("updated_at", "")),
      ),
      reverse=True,
    )
    for item in sorted_items:
      name = str(item.get(key_name, "")).strip()
      value = str(item.get(value_name, "")).strip()
      if name and value:
        result.append(f"{name}={value}")
      if len(result) >= limit:
        break
    return result

  @staticmethod
  def _pick_topic_entries(items: tuple[dict, ...], limit: int) -> list[str]:
    sorted_items = sorted(
      [
        item for item in items
        if str(item.get("topic", "")).strip()
      ],
      key=lambda item: (
        int(item.get("mention_count", 0) or 0),
        _safe_float(item.get("confidence"), 0.0),
        str(item.get("updated_at", "") or item.get("last_seen_at", "")),
      ),
      reverse=True,
    )
    result: list[str] = []
    for item in sorted_items[:limit]:
      topic = str(item.get("topic", "")).strip()
      count = int(item.get("mention_count", 0) or 0)
      if not topic:
        continue
      result.append(f"{topic}×{count}" if count > 1 else topic)
    return result

  @staticmethod
  def _pick_sensitive_entries(items: tuple[dict, ...], limit: int) -> list[str]:
    severity_rank = {"high": 3, "medium": 2, "low": 1}
    sorted_items = sorted(
      [
        item for item in items
        if str(item.get("topic", "")).strip()
      ],
      key=lambda item: (
        severity_rank.get(str(item.get("severity", "")).strip().lower(), 0),
        str(item.get("updated_at", "")),
      ),
      reverse=True,
    )
    result: list[str] = []
    for item in sorted_items[:limit]:
      topic = str(item.get("topic", "")).strip()
      reason = str(item.get("reason", "")).strip()
      severity = str(item.get("severity", "")).strip()
      if reason:
        result.append(f"{topic}（{reason}）")
      elif severity:
        result.append(f"{topic}（{severity}）")
      else:
        result.append(topic)
    return result

  _QUESTION_FRAGMENT_RE = re.compile(
    r"^(什么来着|叫啥|谁来着|啥来着|哪个来着|怎么称呼|叫什么|你叫啥|咋称呼|谁啊|是谁)$"
  )
  _SERIALIZED_DICT_RE = re.compile(r"^\{.*\}$")

  @classmethod
  def _clean_alias_list(cls, raw: tuple[str, ...]) -> list[str]:
    """过滤脏数据和问句片段，只保留有效的名字/昵称。"""
    result: list[str] = []
    seen: set[str] = set()
    for item in raw:
      text = str(item or "").strip()
      if not text or text in seen:
        continue
      if cls._SERIALIZED_DICT_RE.match(text):
        continue
      if cls._QUESTION_FRAGMENT_RE.match(text):
        continue
      if len(text) > 30:
        continue
      seen.add(text)
      result.append(text)
    return result

  @staticmethod
  def _sort_items(items: tuple[dict, ...], text_key: str, prefer_recent: bool = False) -> list[dict]:
    return sorted(
      [
        item for item in items
        if str(item.get(text_key, "")).strip()
      ],
      key=lambda item: (
        _safe_float(item.get("freshness"), 0.0) if prefer_recent else _safe_float(item.get("stability_score"), _stability_score(str(item.get(text_key, "")))),
        _safe_float(item.get("confidence"), 0.0),
        _safe_float(item.get("stability_score"), _stability_score(str(item.get(text_key, "")))) if prefer_recent else _safe_float(item.get("freshness"), 0.0),
        str(item.get("updated_at", "")),
      ),
      reverse=True,
    )
