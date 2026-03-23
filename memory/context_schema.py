"""
结构化记忆/上下文 schema

这层不负责具体存储或检索，只定义项目内统一使用的数据形状。
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional


def _now_iso() -> str:
  return datetime.now().isoformat()


def _to_tuple_dicts(items: Optional[list[dict[str, Any]] | tuple[dict[str, Any], ...]]) -> tuple[dict[str, Any], ...]:
  if not items:
    return ()
  return tuple(dict(item) for item in items if isinstance(item, dict))


def _to_tuple_str(items: Optional[list[str] | tuple[str, ...]]) -> tuple[str, ...]:
  if not items:
    return ()
  return tuple(str(item) for item in items if str(item).strip())


_USER_RECENT_HINTS = re.compile(
  r"最近|近两天|近几天|当前|今天|明天|这周|本周|刚刚|现在|正在|"
  r"忙着|准备|打算|计划|周报|面试|考试|考研|搬家|加班|下班|"
  r"赶项目|赶作业|等结果|等通知|等快递|凌晨|今晚|今早|今晨"
)
_USER_STABLE_HINTS = re.compile(
  r"喜欢|最喜欢|习惯|经常|总是|一直|通常|常常|名字叫|姓|职业|"
  r"工作是|做.?工作的|从事|白噪音|入睡|追看|关注.{0,8}年"
)


def _normalize_identity(
  identity: Optional[dict[str, Any]],
  fallback_nicknames: tuple[str, ...] = (),
) -> dict[str, Any]:
  source = dict(identity or {})
  names = _to_tuple_str(source.get("names"))
  nicknames = _to_tuple_str(source.get("nicknames")) or fallback_nicknames
  preferred_address = str(source.get("preferred_address", "")).strip()
  if not preferred_address and nicknames:
    preferred_address = nicknames[-1]

  occupation = source.get("occupation")
  if isinstance(occupation, dict):
    normalized_occupation = {
      key: value for key, value in occupation.items()
      if value not in (None, "", [])
    }
  else:
    occupation_text = str(occupation or "").strip()
    normalized_occupation = {"value": occupation_text} if occupation_text else {}

  result = {
    "names": names,
    "nicknames": nicknames,
    "preferred_address": preferred_address,
    "occupation": normalized_occupation,
  }
  return {
    key: value for key, value in result.items()
    if value not in ((), "", {}, None)
  }


def _upgrade_legacy_relationship_state(
  relationship_state: Optional[dict[str, Any]],
  preferred_address: str = "",
) -> dict[str, Any]:
  state = dict(relationship_state or {})
  warmth = state.pop("warmth", None)
  tease_ok = state.pop("tease_ok", None)
  care_ok = state.pop("care_ok", None)

  if warmth not in (None, "") and state.get("trust") in (None, ""):
    state["trust"] = warmth
  if tease_ok not in (None, "") and state.get("tease_threshold") in (None, ""):
    state["tease_threshold"] = tease_ok
  if care_ok not in (None, "") and state.get("care_threshold") in (None, ""):
    state["care_threshold"] = care_ok
  if preferred_address and state.get("preferred_address") in (None, ""):
    state["preferred_address"] = preferred_address
  if state.get("public_ack_count") not in (None, "") and state.get("publicly_acknowledged") in (None, ""):
    state["publicly_acknowledged"] = bool(state.get("public_ack_count"))

  return {
    key: value for key, value in state.items()
    if value not in (None, "", [])
  }


def _is_recent_user_fact(item: dict[str, Any]) -> bool:
  text = str(item.get("fact", "")).strip()
  if not text:
    return False
  stable_hit = bool(_USER_STABLE_HINTS.search(text))
  recent_hit = bool(_USER_RECENT_HINTS.search(text))
  if recent_hit and not stable_hit:
    return True
  ttl_days = item.get("ttl_days")
  try:
    ttl_value = int(ttl_days) if ttl_days not in (None, "") else 0
  except (TypeError, ValueError):
    ttl_value = 0
  return ttl_value > 0 and ttl_value <= 14 and not stable_hit


def _legacy_topic_profile(derived_features: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
  topic_entries: list[dict[str, Any]] = []
  for item in derived_features:
    name = str(item.get("name", "")).strip().lower()
    value = str(item.get("value", "")).strip()
    if not value:
      continue
    if "topic" not in name:
      continue
    topic_entries.append({
      "topic": value,
      "mention_count": 1,
      "confidence": float(item.get("confidence", 0.6) or 0.6),
    })
  return tuple(topic_entries)


@dataclass(frozen=True)
class UserMemoryRecord:
  """单个用户的结构化记忆真相源"""
  viewer_id: str
  identity: dict[str, Any] = field(default_factory=dict)
  stable_facts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  recent_state: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  topic_profile: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  relationship_state: dict[str, Any] = field(default_factory=dict)
  callbacks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  open_threads: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  sensitive_topics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  cooldowns: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  legacy_sources: tuple[str, ...] = field(default_factory=tuple)
  created_at: str = field(default_factory=_now_iso)
  updated_at: str = field(default_factory=_now_iso)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "UserMemoryRecord":
    legacy_nicknames = _to_tuple_str(data.get("nicknames"))
    identity = _normalize_identity(data.get("identity"), fallback_nicknames=legacy_nicknames)
    relationship_state = _upgrade_legacy_relationship_state(
      data.get("relationship_state"),
      preferred_address=str(identity.get("preferred_address", "")),
    )

    if (
      "stable_facts" in data
      or "recent_state" in data
      or "topic_profile" in data
      or "open_threads" in data
      or "sensitive_topics" in data
      or "identity" in data
    ):
      stable_facts = _to_tuple_dicts(data.get("stable_facts"))
      recent_state = _to_tuple_dicts(data.get("recent_state"))
      topic_profile = _to_tuple_dicts(data.get("topic_profile"))
      callbacks = _to_tuple_dicts(data.get("callbacks"))
      open_threads = _to_tuple_dicts(data.get("open_threads"))
      sensitive_topics = _to_tuple_dicts(data.get("sensitive_topics"))
    else:
      legacy_facts = _to_tuple_dicts(data.get("hard_facts"))
      stable_facts = tuple(item for item in legacy_facts if not _is_recent_user_fact(item))
      recent_state = tuple(item for item in legacy_facts if _is_recent_user_fact(item))
      legacy_features = _to_tuple_dicts(data.get("derived_features"))
      topic_profile = _legacy_topic_profile(legacy_features)
      callbacks = _to_tuple_dicts(data.get("callbacks"))
      open_threads = ()
      sensitive_topics = ()

    return cls(
      viewer_id=str(data.get("viewer_id", "")),
      identity=identity,
      stable_facts=stable_facts,
      recent_state=recent_state,
      topic_profile=topic_profile,
      relationship_state=relationship_state,
      callbacks=callbacks,
      open_threads=open_threads,
      sensitive_topics=sensitive_topics,
      cooldowns=_to_tuple_dicts(data.get("cooldowns")),
      legacy_sources=_to_tuple_str(data.get("legacy_sources")),
      created_at=str(data.get("created_at", _now_iso())),
      updated_at=str(data.get("updated_at", _now_iso())),
    )


@dataclass(frozen=True)
class SelfMemoryRecord:
  """主播自己的结构化记忆"""
  self_said: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  commitments: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  self_threads: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  stable_preferences: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  legacy_sources: tuple[str, ...] = field(default_factory=tuple)
  created_at: str = field(default_factory=_now_iso)
  updated_at: str = field(default_factory=_now_iso)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "SelfMemoryRecord":
    return cls(
      self_said=_to_tuple_dicts(data.get("self_said")),
      commitments=_to_tuple_dicts(data.get("commitments")),
      self_threads=_to_tuple_dicts(data.get("self_threads")),
      stable_preferences=_to_tuple_dicts(data.get("stable_preferences")),
      legacy_sources=_to_tuple_str(data.get("legacy_sources")),
      created_at=str(data.get("created_at", _now_iso())),
      updated_at=str(data.get("updated_at", _now_iso())),
    )


@dataclass(frozen=True)
class PersonaSpecRecord:
  """角色设定档，不属于 memory，但属于结构化上下文源"""
  persona: str
  items: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  loaded_from: tuple[str, ...] = field(default_factory=tuple)
  updated_at: str = field(default_factory=_now_iso)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "PersonaSpecRecord":
    return cls(
      persona=str(data.get("persona", "")),
      items=_to_tuple_dicts(data.get("items")),
      loaded_from=_to_tuple_str(data.get("loaded_from")),
      updated_at=str(data.get("updated_at", _now_iso())),
    )


@dataclass(frozen=True)
class CorpusEntry:
  """结构化语料项"""
  corpus_id: str
  kind: str
  text: str
  style_tags: tuple[str, ...] = field(default_factory=tuple)
  scene_tags: tuple[str, ...] = field(default_factory=tuple)
  constraints: tuple[str, ...] = field(default_factory=tuple)
  quality_score: float = 0.5
  source: str = ""
  enabled: bool = True
  updated_at: str = field(default_factory=_now_iso)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "CorpusEntry":
    return cls(
      corpus_id=str(data.get("corpus_id", "")),
      kind=str(data.get("kind", "")),
      text=str(data.get("text", "")),
      style_tags=_to_tuple_str(data.get("style_tags")),
      scene_tags=_to_tuple_str(data.get("scene_tags")),
      constraints=_to_tuple_str(data.get("constraints")),
      quality_score=float(data.get("quality_score", 0.5) or 0.5),
      source=str(data.get("source", "")),
      enabled=bool(data.get("enabled", True)),
      updated_at=str(data.get("updated_at", _now_iso())),
    )


@dataclass(frozen=True)
class ExternalKnowledgeEntry:
  """结构化外部知识项"""
  knowledge_id: str
  topic: str
  category: str
  summary: str
  facts: tuple[dict[str, Any], ...] = field(default_factory=tuple)
  sources: tuple[str, ...] = field(default_factory=tuple)
  tags: tuple[str, ...] = field(default_factory=tuple)
  usage_rules: tuple[str, ...] = field(default_factory=tuple)
  reliability: float = 0.5
  enabled: bool = True
  updated_at: str = field(default_factory=_now_iso)
  expires_at: str = ""

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "ExternalKnowledgeEntry":
    return cls(
      knowledge_id=str(data.get("knowledge_id", "")),
      topic=str(data.get("topic", "")),
      category=str(data.get("category", "")),
      summary=str(data.get("summary", "")),
      facts=_to_tuple_dicts(data.get("facts")),
      sources=_to_tuple_str(data.get("sources")),
      tags=_to_tuple_str(data.get("tags")),
      usage_rules=_to_tuple_str(data.get("usage_rules")),
      reliability=float(data.get("reliability", 0.5) or 0.5),
      enabled=bool(data.get("enabled", True)),
      updated_at=str(data.get("updated_at", _now_iso())),
      expires_at=str(data.get("expires_at", "")),
    )


@dataclass(frozen=True)
class CompiledMemoryContext:
  """MemoryCompiler 输出的结构化上下文"""
  user_memory_lines: tuple[str, ...] = field(default_factory=tuple)
  self_memory_lines: tuple[str, ...] = field(default_factory=tuple)
  persona_lines: tuple[str, ...] = field(default_factory=tuple)
  corpus_lines: tuple[str, ...] = field(default_factory=tuple)
  knowledge_lines: tuple[str, ...] = field(default_factory=tuple)

  def to_prompt_blocks(self) -> str:
    sections: list[str] = []
    if self.user_memory_lines:
      sections.append("【用户记忆】\n" + "\n".join(f"- {line}" for line in self.user_memory_lines))
    if self.self_memory_lines:
      sections.append("【自我记忆】\n" + "\n".join(f"- {line}" for line in self.self_memory_lines))
    if self.persona_lines:
      sections.append("【角色设定档】\n" + "\n".join(f"- {line}" for line in self.persona_lines))
    if self.corpus_lines:
      sections.append("【可用语料参考】\n" + "\n".join(f"- {line}" for line in self.corpus_lines))
    if self.knowledge_lines:
      sections.append("【外部知识参考】\n" + "\n".join(f"- {line}" for line in self.knowledge_lines))
    return "\n\n".join(sections)
