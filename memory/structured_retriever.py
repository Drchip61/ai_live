"""
structured 主检索器

以 JSON 结构化真相源为主库，并维护一组面向检索的 Chroma 投影索引。
"""

from __future__ import annotations

import hashlib
from typing import Optional, Union

from .config import EmbeddingConfig, StructuredContextConfig
from .context_schema import (
  CompiledMemoryContext,
  CorpusEntry,
  ExternalKnowledgeEntry,
  PersonaSpecRecord,
  SelfMemoryRecord,
  UserMemoryRecord,
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

  def rebuild_all(self) -> None:
    self.rebuild_user_indexes()
    self.rebuild_self_said_indexes()
    self.rebuild_self_thread_index()
    self.rebuild_persona_index()
    self.rebuild_corpus_index()
    self.rebuild_knowledge_index()

  def rebuild_user_indexes(self) -> None:
    self._user_fact_index.clear()
    self._user_callback_index.clear()
    if self._user_memory_store is None:
      return
    for record in self._user_memory_store.all().values():
      self._upsert_user_record(record)

  def rebuild_user_record(self, viewer_id: str) -> None:
    self._user_fact_index.delete_where({"viewer_id": viewer_id})
    self._user_callback_index.delete_where({"viewer_id": viewer_id})
    if self._user_memory_store is None:
      return
    record = self._user_memory_store.get(viewer_id)
    if record is not None:
      self._upsert_user_record(record)

  def rebuild_self_said_indexes(self) -> None:
    self._self_said_index.clear()
    self._self_commitment_index.clear()
    if self._self_memory_store is None:
      return
    self_record = self._self_memory_store.get()
    self._upsert_self_said(self_record)
    self._upsert_commitments(self_record)

  def rebuild_self_thread_index(self) -> None:
    self._self_thread_index.clear()
    if self._self_memory_store is None:
      return
    self_record = self._self_memory_store.get()
    self._upsert_self_threads(self_record)

  def rebuild_persona_index(self) -> None:
    self._persona_index.clear()
    if self._persona_spec_store is None:
      return
    record = self._persona_spec_store.get()
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
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
    self._upsert_docs(self._persona_index, doc_ids, contents, metadatas)

  def rebuild_corpus_index(self) -> None:
    self._corpus_index.clear()
    if self._corpus_store is None:
      return
    entries = self._corpus_store.list_enabled()
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
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
    self._upsert_docs(self._corpus_index, doc_ids, contents, metadatas)

  def rebuild_knowledge_index(self) -> None:
    self._knowledge_index.clear()
    if self._external_knowledge_store is None:
      return
    entries = self._external_knowledge_store.list_enabled()
    doc_ids: list[str] = []
    contents: list[str] = []
    metadatas: list[dict] = []
    for entry in entries:
      head = entry.topic or entry.category
      content = f"{head} {entry.summary}".strip()
      if not content:
        continue
      doc_ids.append(_stable_id("knowledge", entry.knowledge_id, content))
      contents.append(content)
      metadatas.append({
        "topic": entry.topic,
        "category": entry.category,
        "display_text": f"{head}：{entry.summary}" if head and entry.summary else content,
        "reliability": entry.reliability,
      })
    self._upsert_docs(self._knowledge_index, doc_ids, contents, metadatas)

  def compile_prompt_context(
    self,
    query: Union[str, list[str]],
    viewer_ids: Optional[list[str]] = None,
    include_corpus: bool = False,
    include_external_knowledge: bool = False,
  ) -> str:
    query_text = self._normalize_query(query)
    context = CompiledMemoryContext(
      user_memory_lines=tuple(self._build_user_lines(query_text, viewer_ids)),
      self_memory_lines=tuple(self._build_self_lines(query_text)),
      persona_lines=tuple(self._build_persona_lines(query_text)),
      corpus_lines=tuple(self._build_corpus_lines(query_text)) if include_corpus else (),
      knowledge_lines=tuple(self._build_knowledge_lines(query_text)) if include_external_knowledge else (),
    )
    return context.to_prompt_blocks()

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

  def _upsert_user_record(self, record: UserMemoryRecord) -> None:
    identity = record.identity or {}
    nicknames = tuple(identity.get("nicknames", ()))
    nickname = str(identity.get("preferred_address", "")).strip() or (nicknames[-1] if nicknames else record.viewer_id)

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

    self._upsert_docs(self._user_fact_index, fact_ids, fact_contents, fact_metadatas)
    self._upsert_docs(self._user_callback_index, callback_ids, callback_contents, callback_metadatas)

  def _upsert_self_said(self, self_record: SelfMemoryRecord) -> None:
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
    self._upsert_docs(self._self_said_index, doc_ids, contents, metadatas)

  def _upsert_commitments(self, self_record: SelfMemoryRecord) -> None:
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
    self._upsert_docs(self._self_commitment_index, doc_ids, contents, metadatas)

  def _upsert_self_threads(self, self_record: SelfMemoryRecord) -> None:
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
    self._upsert_docs(self._self_thread_index, doc_ids, contents, metadatas)

  @staticmethod
  def _upsert_docs(index: VectorStore, doc_ids: list[str], contents: list[str], metadatas: list[dict]) -> None:
    if doc_ids:
      index.add_batch(doc_ids, contents, metadatas)

  @staticmethod
  def _normalize_query(query: Union[str, list[str]]) -> str:
    if isinstance(query, list):
      parts = [str(item).strip() for item in query if str(item).strip()]
      return " ".join(parts)
    return str(query or "").strip()

  def _build_user_lines(self, query_text: str, viewer_ids: Optional[list[str]]) -> list[str]:
    if self._user_memory_store is None:
      return []
    picked_viewers: list[str] = []
    for viewer_id in viewer_ids or []:
      normalized = str(viewer_id).strip()
      if normalized and normalized not in picked_viewers:
        picked_viewers.append(normalized)
      if len(picked_viewers) >= self._config.max_viewers:
        break
    if not picked_viewers:
      return []

    lines = ["使用原则：最多只轻轻打一张关系牌，不要背档案式复述历史。"]
    for viewer_id in picked_viewers:
      record = self._user_memory_store.get(viewer_id)
      if record is None:
        continue
      identity = record.identity or {}
      nicknames = tuple(identity.get("nicknames", ()))
      names = tuple(identity.get("names", ()))
      preferred_address = str(identity.get("preferred_address", "")).strip()
      nickname = preferred_address or (nicknames[-1] if nicknames else record.viewer_id)
      lines.append(f"当前关注对象：{nickname}")

      sensitive_topics = self._pick_sensitive_entries(
        record.sensitive_topics,
        limit=self._config.user_sensitive_top_k,
      )
      if sensitive_topics:
        lines.append(f"{nickname} 的边界提醒：" + "；".join(sensitive_topics))

      identity_parts: list[str] = []
      if preferred_address:
        identity_parts.append(f"建议称呼={preferred_address}")
      if names:
        identity_parts.append("名字线索=" + "/".join(names[:2]))
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
      if last_dialogue_stop and not record.open_threads:
        state_parts.append(f"上次停在={last_dialogue_stop}")
      if state_parts:
        lines.append(f"{nickname} 的关系状态：" + "，".join(state_parts))

      facts = self._search_user_memories(
        self._user_fact_index,
        record,
        query_text=query_text,
        top_k=self._config.user_fact_top_k,
        fallback_items=record.stable_facts,
        text_key="fact",
      )
      if facts:
        lines.append(f"{nickname} 的稳定事实：" + "；".join(facts))

      recent_state = self._fallback_texts(
        record.recent_state,
        "fact",
        self._config.user_recent_state_top_k,
        prefer_recent=True,
      )
      if recent_state:
        lines.append(f"{nickname} 最近在忙/最近状态：" + "；".join(recent_state))

      topic_lines = self._pick_topic_entries(record.topic_profile, self._config.user_topic_top_k)
      if topic_lines:
        lines.append(f"{nickname} 常聊话题：" + "，".join(topic_lines))

      callbacks = self._search_user_memories(
        self._user_callback_index,
        record,
        query_text=query_text,
        top_k=self._config.user_callback_top_k,
        fallback_items=record.callbacks,
        text_key="hook",
      )
      if callbacks:
        lines.append(f"{nickname} 的历史梗/回钩线索：" + "；".join(callbacks))

      open_threads = self._fallback_texts(
        record.open_threads,
        "thread",
        self._config.user_open_thread_top_k,
        prefer_recent=True,
      )
      if open_threads:
        lines.append(f"{nickname} 上次对话停在：" + "；".join(open_threads))
    return lines

  def _build_self_lines(self, query_text: str) -> list[str]:
    if self._self_memory_store is None:
      return []
    record = self._self_memory_store.get()
    lines: list[str] = []

    self_said = self._search_or_fallback(
      self._self_said_index,
      query_text=query_text,
      top_k=self._config.self_said_top_k,
      fallback_items=record.self_said,
      text_key="text",
    )
    if self_said:
      lines.append("和当前话题相关的我说过：" + "；".join(self_said))

    commitments = self._search_or_fallback(
      self._self_commitment_index,
      query_text=query_text,
      top_k=self._config.self_commitment_top_k,
      fallback_items=record.commitments,
      text_key="text",
    )
    if commitments:
      lines.append("仍在延续的承诺/话头：" + "；".join(commitments))

    threads = self._search_or_fallback(
      self._self_thread_index,
      query_text=query_text,
      top_k=self._config.self_thread_top_k,
      fallback_items=record.self_threads,
      text_key="text",
    )
    if threads:
      lines.append("可续接的旧线头：" + "；".join(threads))

    preferences = self._sort_items(record.stable_preferences, "text")[:2]
    preference_texts = [str(item.get("text", "")).strip() for item in preferences if str(item.get("text", "")).strip()]
    if preference_texts:
      lines.append("较稳定的表达偏好：" + "；".join(preference_texts))
    return lines

  def _build_persona_lines(self, query_text: str) -> list[str]:
    if self._persona_spec_store is None:
      return []
    record = self._persona_spec_store.get()
    if query_text:
      lines = self._search_texts(self._persona_index, query_text, top_k=self._config.persona_top_k)
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

  def _build_corpus_lines(self, query_text: str) -> list[str]:
    if self._corpus_store is None:
      return []
    if query_text:
      lines = self._search_texts(self._corpus_index, query_text, top_k=self._config.corpus_top_k)
      if lines:
        return lines
    return [entry.text for entry in self._corpus_store.list_enabled()[:self._config.corpus_top_k]]

  def _build_knowledge_lines(self, query_text: str) -> list[str]:
    if self._external_knowledge_store is None:
      return []
    if query_text:
      lines = self._search_texts(self._knowledge_index, query_text, top_k=self._config.knowledge_top_k)
      if lines:
        return lines
    result: list[str] = []
    for entry in self._external_knowledge_store.list_enabled()[:self._config.knowledge_top_k]:
      head = entry.topic or entry.category
      if entry.summary:
        result.append(f"{head}：{entry.summary}" if head else entry.summary)
    return result

  def _search_user_memories(
    self,
    index: VectorStore,
    record: UserMemoryRecord,
    query_text: str,
    top_k: int,
    fallback_items: tuple[dict, ...],
    text_key: str,
  ) -> list[str]:
    if query_text:
      lines = self._search_texts(
        index,
        query_text,
        top_k=top_k,
        where={"viewer_id": record.viewer_id},
      )
      if lines:
        return lines
    return self._fallback_texts(fallback_items, text_key, top_k)

  def _search_or_fallback(
    self,
    index: VectorStore,
    query_text: str,
    top_k: int,
    fallback_items: tuple[dict, ...],
    text_key: str,
  ) -> list[str]:
    if query_text:
      lines = self._search_texts(index, query_text, top_k=top_k)
      if lines:
        return lines
    return self._fallback_texts(fallback_items, text_key, top_k)

  def _search_texts(
    self,
    index: VectorStore,
    query_text: str,
    top_k: int,
    where: Optional[dict] = None,
  ) -> list[str]:
    if not query_text:
      return []
    picked: list[str] = []
    seen: set[str] = set()
    for doc, _score in index.search(query_text, top_k=top_k, where=where):
      meta = doc.metadata or {}
      text = str(meta.get("display_text", "")).strip() or str(doc.page_content or "").strip()
      if not text or text in seen:
        continue
      seen.add(text)
      picked.append(text)
      if len(picked) >= top_k:
        break
    return picked

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
