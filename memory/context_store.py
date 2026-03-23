"""
结构化上下文存储

这里的目标不是替代向量库，而是提供稳定的结构化真相源。
"""

import json
import logging
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Optional

from .context_schema import (
  CorpusEntry,
  ExternalKnowledgeEntry,
  PersonaSpecRecord,
  SelfMemoryRecord,
  UserMemoryRecord,
)

logger = logging.getLogger(__name__)

_TEXT_NORMALIZE_RE = re.compile(r"[\s，。！？、,.!?:：;；“”\"'（）()《》<>【】\[\]~～…·`]+")
_GENERIC_SUMMARY_PATTERNS = (
  re.compile(r"^我(?:最近)?与多位观众"),
  re.compile(r"^我在与观众(?:们)?的互动中"),
  re.compile(r"^我和(?:多位|不少|很多)?观众"),
  re.compile(r"^在最近的直播中"),
  re.compile(r"^最近(?:这段时间)?(?:的)?互动"),
)
_SOURCE_PRIORITY = {
  "viewer_summary_extract": 6,
  "stance_extraction": 6,
  "self_commitment": 6,
  "summary_rollup": 5,
  "legacy_temporary": 4,
  "archive_temporary": 4,
  "legacy_stance": 3,
  "legacy_viewer": 3,
  "legacy_preference": 3,
  "archive_viewer": 2,
  "archive_stance": 1,
  "legacy_summary": 1,
  "archive_summary": 1,
}


def _now_iso() -> str:
  return datetime.now().isoformat()


def _unique_keep_order(items: list[str]) -> tuple[str, ...]:
  seen: set[str] = set()
  result: list[str] = []
  for item in items:
    text = str(item).strip()
    if not text or text in seen:
      continue
    seen.add(text)
    result.append(text)
  return tuple(result)


def _normalize_text_for_match(text: str) -> str:
  return _TEXT_NORMALIZE_RE.sub("", str(text or "").strip()).lower()


def _entry_priority(item: dict, semantic_mode: str, text_key: str) -> float:
  source = str(item.get("source", "") or item.get("source_layer", "")).strip()
  confidence = float(item.get("confidence", 0.0) or 0.0)
  freshness = float(item.get("freshness", 0.0) or 0.0)
  text = str(item.get(text_key, "")).strip()
  score = float(_SOURCE_PRIORITY.get(source, 0)) * 100
  score += confidence * 10
  score += freshness * 5
  if semantic_mode == "thread":
    if "temporary" in source:
      score += 20
    if len(text) > 180:
      score -= min(len(text) - 180, 120) / 4
  elif semantic_mode == "state":
    score += freshness * 20
    score += max(0, 30 - float(item.get("ttl_days", 30) or 30)) / 10
  else:
    score -= min(len(text), 180) / 1000
  return score


def _is_near_duplicate(left: str, right: str, semantic_mode: str) -> bool:
  left_norm = _normalize_text_for_match(left)
  right_norm = _normalize_text_for_match(right)
  if not left_norm or not right_norm:
    return False
  if left_norm == right_norm:
    return True
  shorter, longer = sorted((left_norm, right_norm), key=len)
  if len(shorter) >= 8 and shorter in longer:
    return True
  if semantic_mode == "thread":
    return False
  ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
  threshold = {
    "fact": 0.88,
    "state": 0.9,
    "callback": 0.9,
    "stance": 0.9,
    "commitment": 0.9,
    "preference": 0.9,
  }.get(semantic_mode, 0.95)
  return ratio >= threshold


def _merge_duplicate_items(
  current: dict,
  incoming: dict,
  text_key: str,
  semantic_mode: str,
) -> dict:
  current_score = _entry_priority(current, semantic_mode, text_key)
  incoming_score = _entry_priority(incoming, semantic_mode, text_key)
  prefer_current = current_score >= incoming_score
  merged = dict(current if prefer_current else incoming)
  fallback = incoming if prefer_current else current
  for key, value in fallback.items():
    if key not in merged or merged.get(key) in (None, "", []):
      merged[key] = value
  if "confidence" in current or "confidence" in incoming:
    merged["confidence"] = max(
      float(current.get("confidence", 0.0) or 0.0),
      float(incoming.get("confidence", 0.0) or 0.0),
    )
  if "freshness" in current or "freshness" in incoming:
    merged["freshness"] = max(
      float(current.get("freshness", 0.0) or 0.0),
      float(incoming.get("freshness", 0.0) or 0.0),
    )
  merged.setdefault("created_at", current.get("created_at") or incoming.get("created_at") or _now_iso())
  merged["updated_at"] = _now_iso()
  return merged


def _merge_text_entries(
  existing: tuple[dict, ...],
  incoming: list[dict],
  text_key: str,
  semantic_mode: str = "default",
) -> tuple[dict, ...]:
  merged: dict[str, dict] = {}
  for item in existing:
    key = str(item.get(text_key, "")).strip()
    if key:
      merged[key] = dict(item)

  for item in incoming:
    key = str(item.get(text_key, "")).strip()
    if not key:
      continue
    match_key = key if key in merged else None
    if match_key is None:
      for existing_key, existing_item in merged.items():
        existing_text = str(existing_item.get(text_key, existing_key)).strip()
        if _is_near_duplicate(existing_text, key, semantic_mode):
          match_key = existing_key
          break
    if match_key is None:
      new_item = dict(item)
      new_item.setdefault("created_at", _now_iso())
      new_item.setdefault("updated_at", _now_iso())
      merged[key] = new_item
      continue
    merged[match_key] = _merge_duplicate_items(
      dict(merged[match_key]),
      dict(item),
      text_key=text_key,
      semantic_mode=semantic_mode,
    )
  return tuple(merged.values())


def _merge_named_entries(
  existing: tuple[dict, ...],
  incoming: list[dict],
  key_name: str,
) -> tuple[dict, ...]:
  merged: dict[str, dict] = {}
  for item in existing:
    key = str(item.get(key_name, "")).strip()
    if key:
      merged[key] = dict(item)

  for item in incoming:
    key = str(item.get(key_name, "")).strip()
    if not key:
      continue
    updated = dict(merged.get(key, {}))
    updated.update(item)
    updated.setdefault("created_at", _now_iso())
    updated["updated_at"] = _now_iso()
    merged[key] = updated
  return tuple(merged.values())


def _coerce_float(value, default: Optional[float] = None) -> Optional[float]:
  if value in (None, ""):
    return default
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _merge_identity(existing: dict, incoming: Optional[dict], nickname: str = "") -> dict:
  current = UserMemoryRecord.from_dict({
    "viewer_id": "tmp",
    "identity": existing or {},
  }).identity
  incoming_identity = UserMemoryRecord.from_dict({
    "viewer_id": "tmp",
    "identity": incoming or {},
  }).identity

  names = _unique_keep_order(
    list(current.get("names", ())) + list(incoming_identity.get("names", ()))
  )
  nicknames = _unique_keep_order(
    list(current.get("nicknames", ()))
    + list(incoming_identity.get("nicknames", ()))
    + ([nickname] if nickname else [])
  )
  preferred_address = (
    str(incoming_identity.get("preferred_address", "")).strip()
    or str(current.get("preferred_address", "")).strip()
    or (nicknames[-1] if nicknames else "")
  )

  occupation = dict(current.get("occupation", {}) or {})
  incoming_occupation = dict(incoming_identity.get("occupation", {}) or {})
  if incoming_occupation.get("value"):
    occupation.update(incoming_occupation)

  result = {
    "names": names,
    "nicknames": nicknames,
    "preferred_address": preferred_address,
    "occupation": occupation,
  }
  return {
    key: value for key, value in result.items()
    if value not in ((), "", {}, None)
  }


def _merge_topic_entries(
  existing: tuple[dict, ...],
  incoming: list[dict],
) -> tuple[dict, ...]:
  merged: dict[str, dict] = {}
  for item in existing:
    topic = str(item.get("topic", "")).strip()
    if topic:
      merged[topic] = dict(item)

  for item in incoming:
    topic = str(item.get("topic", "")).strip()
    if not topic:
      continue
    current = dict(merged.get(topic, {}))
    mention_count = int(item.get("mention_count", item.get("count", 1)) or 1)
    updated = {
      "topic": topic,
      "mention_count": int(current.get("mention_count", 0) or 0) + mention_count,
      "confidence": max(
        float(current.get("confidence", 0.0) or 0.0),
        float(item.get("confidence", 0.0) or 0.0),
      ),
      "last_seen_at": str(item.get("last_seen_at", "")).strip() or _now_iso(),
      "created_at": str(current.get("created_at", "")).strip() or _now_iso(),
      "updated_at": _now_iso(),
    }
    if item.get("source"):
      updated["source"] = item.get("source")
    elif current.get("source"):
      updated["source"] = current.get("source")
    merged[topic] = updated
  return tuple(merged.values())


def _merge_relationship_state(
  existing: dict,
  incoming: Optional[dict],
  preferred_address: str = "",
  was_addressed: bool = False,
) -> dict:
  merged = dict(existing or {})
  payload = dict(incoming or {})

  for key in ("familiarity", "trust", "tease_threshold", "care_threshold"):
    incoming_value = _coerce_float(payload.get(key))
    if incoming_value is None:
      continue
    existing_value = _coerce_float(merged.get(key))
    if existing_value is None:
      merged[key] = round(incoming_value, 3)
    else:
      merged[key] = round(existing_value * 0.7 + incoming_value * 0.3, 3)

  for key in ("interaction_style", "address_style", "preferred_address", "last_dialogue_stop"):
    value = payload.get(key)
    if value not in (None, "", []):
      merged[key] = value

  if preferred_address and merged.get("preferred_address") in (None, ""):
    merged["preferred_address"] = preferred_address

  if payload.get("publicly_acknowledged") not in (None, ""):
    merged["publicly_acknowledged"] = bool(payload.get("publicly_acknowledged"))

  if was_addressed:
    merged["publicly_acknowledged"] = True
    merged["public_ack_count"] = int(merged.get("public_ack_count", 0) or 0) + 1
    merged["last_public_ack_at"] = _now_iso()

  merged["updated_at"] = _now_iso()
  return {
    key: value for key, value in merged.items()
    if value not in (None, "", [])
  }


class _JsonStoreBase:
  """JSON 持久化基类"""

  def __init__(self, persist_path: Optional[Path]) -> None:
    self._persist_path = persist_path

  @property
  def persist_path(self) -> Optional[Path]:
    return self._persist_path

  def _load_json(self, default):
    if self._persist_path is None or not self._persist_path.exists():
      return default
    try:
      return json.loads(self._persist_path.read_text(encoding="utf-8"))
    except Exception as e:
      logger.error("读取结构化上下文失败 %s: %s", self._persist_path, e)
      return default

  def _save_json(self, data) -> None:
    if self._persist_path is None:
      return
    try:
      self._persist_path.parent.mkdir(parents=True, exist_ok=True)
      self._persist_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
      )
    except Exception as e:
      logger.error("保存结构化上下文失败 %s: %s", self._persist_path, e)


class UserMemoryStore(_JsonStoreBase):
  """用户结构化记忆库"""

  def __init__(self, persist_path: Optional[Path]) -> None:
    super().__init__(persist_path)
    raw = self._load_json({})
    self._records: dict[str, UserMemoryRecord] = {}
    needs_rewrite = False
    for viewer_id, record in raw.items():
      if not isinstance(record, dict):
        continue
      if not self._looks_like_current_schema(record):
        needs_rewrite = True
      self._records[viewer_id] = UserMemoryRecord.from_dict(record)
    if needs_rewrite:
      self._persist()

  def get(self, viewer_id: str) -> Optional[UserMemoryRecord]:
    return self._records.get(viewer_id)

  def all(self) -> dict[str, UserMemoryRecord]:
    return dict(self._records)

  def record_extract(
    self,
    viewer_id: str,
    nickname: str,
    identity: Optional[dict] = None,
    stable_facts: Optional[list[dict]] = None,
    recent_state: Optional[list[dict]] = None,
    topic_profile: Optional[list[dict]] = None,
    relationship_state: Optional[dict] = None,
    callbacks: Optional[list[dict]] = None,
    open_threads: Optional[list[dict]] = None,
    sensitive_topics: Optional[list[dict]] = None,
    legacy_source: str = "",
    was_addressed: bool = False,
  ) -> UserMemoryRecord:
    record = self._records.get(viewer_id)
    if record is None:
      record = UserMemoryRecord(viewer_id=viewer_id)

    merged_identity = _merge_identity(record.identity, identity, nickname)
    stable_fact_items = _merge_text_entries(record.stable_facts, stable_facts or [], "fact", semantic_mode="fact")
    recent_state_items = _merge_text_entries(record.recent_state, recent_state or [], "fact", semantic_mode="state")
    topic_items = _merge_topic_entries(record.topic_profile, topic_profile or [])
    callback_items = _merge_text_entries(record.callbacks, callbacks or [], "hook", semantic_mode="callback")
    open_thread_items = _merge_text_entries(record.open_threads, open_threads or [], "thread", semantic_mode="callback")
    sensitive_topic_items = _merge_named_entries(record.sensitive_topics, sensitive_topics or [], "topic")
    merged_relationship = _merge_relationship_state(
      record.relationship_state,
      relationship_state,
      preferred_address=str(merged_identity.get("preferred_address", "")),
      was_addressed=was_addressed,
    )

    legacy_sources = list(record.legacy_sources)
    if legacy_source and legacy_source not in legacy_sources:
      legacy_sources.append(legacy_source)

    updated = UserMemoryRecord(
      viewer_id=viewer_id,
      identity=merged_identity,
      stable_facts=stable_fact_items,
      recent_state=recent_state_items,
      topic_profile=topic_items,
      relationship_state=merged_relationship,
      callbacks=callback_items,
      open_threads=open_thread_items,
      sensitive_topics=sensitive_topic_items,
      cooldowns=record.cooldowns,
      legacy_sources=tuple(legacy_sources),
      created_at=record.created_at,
      updated_at=_now_iso(),
    )
    self._records[viewer_id] = updated
    self._persist()
    return updated

  def set_cooldown(
    self,
    viewer_id: str,
    key: str,
    cooldown_until: str,
    reason: str = "",
  ) -> UserMemoryRecord:
    record = self._records.get(viewer_id) or UserMemoryRecord(viewer_id=viewer_id)
    cooldowns = _merge_named_entries(
      record.cooldowns,
      [{
        "key": key,
        "cooldown_until": cooldown_until,
        "reason": reason,
      }],
      "key",
    )
    updated = UserMemoryRecord(
      viewer_id=record.viewer_id,
      identity=record.identity,
      stable_facts=record.stable_facts,
      recent_state=record.recent_state,
      topic_profile=record.topic_profile,
      relationship_state=dict(record.relationship_state),
      callbacks=record.callbacks,
      open_threads=record.open_threads,
      sensitive_topics=record.sensitive_topics,
      cooldowns=cooldowns,
      legacy_sources=record.legacy_sources,
      created_at=record.created_at,
      updated_at=_now_iso(),
    )
    self._records[viewer_id] = updated
    self._persist()
    return updated

  @staticmethod
  def _looks_like_current_schema(record: dict) -> bool:
    return any(
      key in record for key in (
        "identity",
        "stable_facts",
        "recent_state",
        "topic_profile",
        "open_threads",
        "sensitive_topics",
      )
    )

  def debug_state(self) -> dict:
    return {
      "count": len(self._records),
      "viewer_ids": list(self._records.keys()),
      "sample": [record.to_dict() for record in list(self._records.values())[:5]],
    }

  def clear(self) -> None:
    self._records = {}
    self._persist()

  def _persist(self) -> None:
    self._save_json({
      viewer_id: record.to_dict()
      for viewer_id, record in self._records.items()
    })


class SelfMemoryStore(_JsonStoreBase):
  """主播自我记忆库"""

  def __init__(self, persist_path: Optional[Path]) -> None:
    super().__init__(persist_path)
    raw = self._load_json({})
    self._record = SelfMemoryRecord.from_dict(raw if isinstance(raw, dict) else {})

  def get(self) -> SelfMemoryRecord:
    return self._record

  def record_stance(
    self,
    topic: str,
    statement: str,
    response_excerpt: str = "",
    source: str = "stance_extraction",
  ) -> SelfMemoryRecord:
    self_said = _merge_text_entries(
      self._record.self_said,
      [{
        "topic": topic,
        "text": statement,
        "response_excerpt": response_excerpt[:200],
        "source": source,
        "confidence": 0.7,
      }],
      "text",
      semantic_mode="stance",
    )
    commitments = self._record.commitments
    if self._looks_like_commitment(statement):
      commitments = _merge_text_entries(
        self._record.commitments,
        [{
          "text": statement,
          "topic": topic,
          "status": "open",
          "source": source,
        }],
        "text",
        semantic_mode="commitment",
      )

    self._record = SelfMemoryRecord(
      self_said=self_said,
      commitments=commitments,
      self_threads=self._record.self_threads,
      stable_preferences=self._record.stable_preferences,
      legacy_sources=self._merge_legacy_sources(source),
      created_at=self._record.created_at,
      updated_at=_now_iso(),
    )
    self._persist()
    return self._record

  def add_thread_memory(self, text: str, source_layer: str) -> SelfMemoryRecord:
    if not self.should_keep_thread(text, source_layer):
      return self._record
    self_threads = _merge_text_entries(
      self._record.self_threads,
      [{
        "text": text,
        "source_layer": source_layer,
        "status": "legacy_fallback",
      }],
      "text",
      semantic_mode="thread",
    )
    self._record = SelfMemoryRecord(
      self_said=self._record.self_said,
      commitments=self._record.commitments,
      self_threads=self_threads,
      stable_preferences=self._record.stable_preferences,
      legacy_sources=self._merge_legacy_sources(source_layer),
      created_at=self._record.created_at,
      updated_at=_now_iso(),
    )
    self._persist()
    return self._record

  def add_commitment(
    self,
    text: str,
    topic: str = "",
    source: str = "self_commitment",
    status: str = "open",
  ) -> SelfMemoryRecord:
    commitments = _merge_text_entries(
      self._record.commitments,
      [{
        "text": text,
        "topic": topic,
        "source": source,
        "status": status,
      }],
      "text",
      semantic_mode="commitment",
    )
    self._record = SelfMemoryRecord(
      self_said=self._record.self_said,
      commitments=commitments,
      self_threads=self._record.self_threads,
      stable_preferences=self._record.stable_preferences,
      legacy_sources=self._merge_legacy_sources(source),
      created_at=self._record.created_at,
      updated_at=_now_iso(),
    )
    self._persist()
    return self._record

  def debug_state(self) -> dict:
    return self._record.to_dict()

  def clear(self) -> None:
    self._record = SelfMemoryRecord()
    self._persist()

  def _merge_legacy_sources(self, source: str) -> tuple[str, ...]:
    merged = list(self._record.legacy_sources)
    if source and source not in merged:
      merged.append(source)
    return tuple(merged)

  @staticmethod
  def _looks_like_commitment(text: str) -> bool:
    markers = ("我会", "以后我", "下次我", "我答应", "我准备", "我打算", "我之后")
    return any(marker in text for marker in markers)

  @staticmethod
  def should_keep_thread(text: str, source_layer: str) -> bool:
    normalized = str(text).strip()
    if not normalized:
      return False
    layer = str(source_layer or "").strip()
    if "summary" not in layer:
      return True
    if len(normalized) >= 220:
      return False
    return not any(pattern.search(normalized) for pattern in _GENERIC_SUMMARY_PATTERNS)

  def _persist(self) -> None:
    self._save_json(self._record.to_dict())


class PersonaSpecStore(_JsonStoreBase):
  """角色设定档"""

  def __init__(self, persist_path: Optional[Path], persona: str) -> None:
    super().__init__(persist_path)
    raw = self._load_json({})
    if isinstance(raw, dict) and raw.get("persona"):
      self._record = PersonaSpecRecord.from_dict(raw)
    else:
      self._record = PersonaSpecRecord(persona=persona)

  def get(self) -> PersonaSpecRecord:
    return self._record

  def list_sections(self) -> list[str]:
    """返回所有可用的 section 名称（去重、保序）"""
    seen: set[str] = set()
    result: list[str] = []
    for item in self._record.items:
      section = str(item.get("section", "")).strip()
      if section and section not in seen:
        seen.add(section)
        result.append(section)
    return result

  def get_by_sections(self, sections: list[str]) -> list[dict]:
    """按 section 精确匹配检索条目"""
    section_set = set(sections)
    return [
      item for item in self._record.items
      if str(item.get("section", "")).strip() in section_set
    ]

  def load_from_static_dir(self, static_dir: Path) -> PersonaSpecRecord:
    if not static_dir.exists():
      return self._record

    items_by_key: dict[tuple[str, str], dict] = {}
    for item in self._record.items:
      key = (str(item.get("section", "")), str(item.get("text", "")))
      if key[1]:
        items_by_key[key] = dict(item)

    loaded_from = list(self._record.loaded_from)

    for json_file in sorted(static_dir.glob("*.json")):
      try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
      except Exception as e:
        logger.error("加载 persona spec 失败 %s: %s", json_file, e)
        continue
      if json_file.name not in loaded_from:
        loaded_from.append(json_file.name)
      if not isinstance(data, list):
        continue
      for entry in data:
        content = str(entry.get("content", "")).strip()
        if not content:
          continue
        section = str(entry.get("category", "") or json_file.stem)
        items_by_key[(section, content)] = {
          "section": section,
          "text": content,
          "source_file": json_file.name,
          "loaded_at": _now_iso(),
        }

    self._record = PersonaSpecRecord(
      persona=self._record.persona,
      items=tuple(items_by_key.values()),
      loaded_from=tuple(loaded_from),
      updated_at=_now_iso(),
    )
    self._persist()
    return self._record

  def debug_state(self) -> dict:
    return self._record.to_dict()

  def clear(self) -> None:
    self._record = PersonaSpecRecord(persona=self._record.persona)
    self._persist()

  def _persist(self) -> None:
    self._save_json(self._record.to_dict())


class CorpusStore(_JsonStoreBase):
  """结构化语料库"""

  def __init__(self, persist_path: Optional[Path]) -> None:
    super().__init__(persist_path)
    raw = self._load_json([])
    self._entries: list[CorpusEntry] = [
      CorpusEntry.from_dict(item)
      for item in raw
      if isinstance(item, dict)
    ]

  def upsert(self, entry: CorpusEntry) -> CorpusEntry:
    kept = [item for item in self._entries if item.corpus_id != entry.corpus_id]
    kept.append(entry)
    self._entries = kept
    self._persist()
    return entry

  def list_enabled(self) -> list[CorpusEntry]:
    return [entry for entry in self._entries if entry.enabled]

  def list_style_tags(self) -> list[str]:
    """返回所有可用的 style_tags（去重）"""
    tags: set[str] = set()
    for entry in self._entries:
      if entry.enabled:
        tags.update(entry.style_tags)
    return sorted(tags)

  def list_scene_tags(self) -> list[str]:
    """返回所有可用的 scene_tags（去重）"""
    tags: set[str] = set()
    for entry in self._entries:
      if entry.enabled:
        tags.update(entry.scene_tags)
    return sorted(tags)

  def get_by_tags(
    self,
    style_tag: str = "",
    scene_tag: str = "",
    limit: int = 5,
  ) -> list[CorpusEntry]:
    """按 style_tag 和 scene_tag 筛选语料条目"""
    result: list[CorpusEntry] = []
    for entry in self._entries:
      if not entry.enabled:
        continue
      if style_tag and style_tag not in entry.style_tags:
        continue
      if scene_tag and scene_tag not in entry.scene_tags:
        continue
      result.append(entry)
      if len(result) >= limit:
        break
    return result

  def debug_state(self) -> dict:
    return {
      "count": len(self._entries),
      "enabled_count": len(self.list_enabled()),
      "sample": [entry.to_dict() for entry in self._entries[:5]],
    }

  def clear(self) -> None:
    self._entries = []
    self._persist()

  def _persist(self) -> None:
    self._save_json([entry.to_dict() for entry in self._entries])


class ExternalKnowledgeStore(_JsonStoreBase):
  """结构化外部知识库"""

  def __init__(self, persist_path: Optional[Path]) -> None:
    super().__init__(persist_path)
    raw = self._load_json([])
    self._entries: list[ExternalKnowledgeEntry] = [
      ExternalKnowledgeEntry.from_dict(item)
      for item in raw
      if isinstance(item, dict)
    ]

  def upsert(self, entry: ExternalKnowledgeEntry) -> ExternalKnowledgeEntry:
    kept = [item for item in self._entries if item.knowledge_id != entry.knowledge_id]
    kept.append(entry)
    self._entries = kept
    self._persist()
    return entry

  def list_enabled(self) -> list[ExternalKnowledgeEntry]:
    return [entry for entry in self._entries if entry.enabled]

  def list_topics(self) -> list[str]:
    """返回所有已启用条目的 topic 名称列表"""
    return [
      entry.topic for entry in self._entries
      if entry.enabled and entry.topic
    ]

  def get_by_topics(self, topics: list[str]) -> list[ExternalKnowledgeEntry]:
    """按 topic 精确匹配检索已启用的条目"""
    topic_set = set(topics)
    return [
      entry for entry in self._entries
      if entry.enabled and entry.topic in topic_set
    ]

  def debug_state(self) -> dict:
    return {
      "count": len(self._entries),
      "enabled_count": len(self.list_enabled()),
      "sample": [entry.to_dict() for entry in self._entries[:5]],
    }

  def clear(self) -> None:
    self._entries = []
    self._persist()

  def _persist(self) -> None:
    self._save_json([entry.to_dict() for entry in self._entries])
