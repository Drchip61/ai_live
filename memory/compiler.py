"""
结构化记忆编译器

把结构化真相源压缩成可供 prompt 使用的短上下文块。
"""

from dataclasses import dataclass
from typing import Optional

from .context_schema import (
  CompiledMemoryContext,
  CorpusEntry,
  ExternalKnowledgeEntry,
  PersonaSpecRecord,
  SelfMemoryRecord,
  UserMemoryRecord,
  resolve_preferred_address,
)


@dataclass(frozen=True)
class CompilerLimits:
  max_user_facts: int = 3
  max_user_callbacks: int = 2
  max_self_said: int = 3
  max_commitments: int = 2
  max_persona_items: int = 4
  max_corpus_items: int = 3
  max_knowledge_items: int = 3


class MemoryCompiler:
  """只编译 UserMemory + SelfMemory"""

  def __init__(self, limits: CompilerLimits = CompilerLimits()):
    self._limits = limits

  def compile_memory(
    self,
    user_memory: Optional[UserMemoryRecord],
    self_memory: Optional[SelfMemoryRecord],
  ) -> CompiledMemoryContext:
    user_lines = self._compile_user_memory(user_memory)
    self_lines = self._compile_self_memory(self_memory)
    return CompiledMemoryContext(
      user_memory_lines=tuple(user_lines),
      self_memory_lines=tuple(self_lines),
    )

  def _compile_user_memory(self, user_memory: Optional[UserMemoryRecord]) -> list[str]:
    if user_memory is None:
      return []

    lines: list[str] = []
    identity = user_memory.identity or {}
    nicknames = tuple(identity.get("nicknames", ()))
    names = tuple(identity.get("names", ()))
    nickname = resolve_preferred_address(
      identity,
      fallback_nicknames=nicknames,
      raw_aliases=(user_memory.viewer_id,),
      fallback=user_memory.viewer_id,
    )
    lines.append(f"当前关注对象：{nickname}")

    state = user_memory.relationship_state or {}
    familiarity = state.get("familiarity")
    trust = state.get("trust")
    tease_threshold = state.get("tease_threshold")
    interaction_style = state.get("interaction_style")
    address_style = state.get("address_style")

    state_parts: list[str] = []
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
    if state_parts:
      lines.append("关系状态：" + "，".join(state_parts))

    identity_parts: list[str] = []
    occupation = identity.get("occupation", {}) or {}
    occupation_value = str(occupation.get("value", "")).strip() if isinstance(occupation, dict) else ""
    if names:
      identity_parts.append("名字线索=" + "/".join(names[:2]))
    if occupation_value:
      identity_parts.append(f"职业={occupation_value}")
    if identity_parts:
      lines.append("身份信息：" + "，".join(identity_parts))

    facts = self._pick_entries(user_memory.stable_facts, "fact", self._limits.max_user_facts)
    if facts:
      lines.append("稳定事实：" + "；".join(facts))

    recent_state = self._pick_entries(user_memory.recent_state, "fact", 2)
    if recent_state:
      lines.append("最近状态：" + "；".join(recent_state))

    topics = []
    for item in user_memory.topic_profile[:3]:
      topic = str(item.get("topic", "")).strip()
      count = int(item.get("mention_count", 0) or 0)
      if topic:
        topics.append(f"{topic}×{count}" if count > 1 else topic)
    if topics:
      lines.append("常聊话题：" + "，".join(topics))

    callbacks = self._pick_entries(user_memory.callbacks, "hook", self._limits.max_user_callbacks)
    if callbacks:
      lines.append("历史梗/回钩线索：" + "；".join(callbacks))

    open_threads = self._pick_entries(user_memory.open_threads, "thread", 2)
    if open_threads:
      lines.append("上次对话停在：" + "；".join(open_threads))

    sensitive_topics = []
    for item in user_memory.sensitive_topics[:2]:
      topic = str(item.get("topic", "")).strip()
      reason = str(item.get("reason", "")).strip()
      if topic:
        sensitive_topics.append(f"{topic}（{reason}）" if reason else topic)
    if sensitive_topics:
      lines.append("不要主动碰的话题：" + "；".join(sensitive_topics))

    return lines

  def _compile_self_memory(self, self_memory: Optional[SelfMemoryRecord]) -> list[str]:
    if self_memory is None:
      return []

    lines: list[str] = []
    self_said = self._pick_entries(self_memory.self_said, "text", self._limits.max_self_said)
    if self_said:
      lines.append("我之前说过：" + "；".join(self_said))

    commitments = self._pick_entries(self_memory.commitments, "text", self._limits.max_commitments)
    if commitments:
      lines.append("我还在延续的承诺/话头：" + "；".join(commitments))

    preferences = self._pick_entries(self_memory.stable_preferences, "text", 2)
    if preferences:
      lines.append("我较稳定的表达偏好：" + "；".join(preferences))

    return lines

  @staticmethod
  def _pick_entries(entries: tuple[dict, ...], key: str, limit: int) -> list[str]:
    picked: list[str] = []
    for item in entries:
      text = str(item.get(key, "")).strip()
      if text:
        picked.append(text)
      if len(picked) >= limit:
        break
    return picked


class ContextCompiler:
  """把 memory + persona + corpus + knowledge 合并成最终结构化上下文"""

  def __init__(self, limits: CompilerLimits = CompilerLimits()):
    self._limits = limits

  def compile_context(
    self,
    memory_context: CompiledMemoryContext,
    persona_spec: Optional[PersonaSpecRecord] = None,
    corpus_entries: Optional[list[CorpusEntry]] = None,
    knowledge_entries: Optional[list[ExternalKnowledgeEntry]] = None,
  ) -> CompiledMemoryContext:
    persona_lines = self._compile_persona(persona_spec)
    corpus_lines = self._compile_corpus(corpus_entries or [])
    knowledge_lines = self._compile_knowledge(knowledge_entries or [])
    return CompiledMemoryContext(
      user_memory_lines=memory_context.user_memory_lines,
      self_memory_lines=memory_context.self_memory_lines,
      persona_lines=tuple(persona_lines),
      corpus_lines=tuple(corpus_lines),
      knowledge_lines=tuple(knowledge_lines),
    )

  def _compile_persona(self, persona_spec: Optional[PersonaSpecRecord]) -> list[str]:
    if persona_spec is None:
      return []
    lines: list[str] = []
    for item in persona_spec.items[:self._limits.max_persona_items]:
      section = str(item.get("section", "")).strip()
      text = str(item.get("text", "")).strip()
      if not text:
        continue
      if section:
        lines.append(f"{section}：{text}")
      else:
        lines.append(text)
    return lines

  def _compile_corpus(self, entries: list[CorpusEntry]) -> list[str]:
    lines: list[str] = []
    for entry in entries[:self._limits.max_corpus_items]:
      tags = "/".join(entry.style_tags[:2]) if entry.style_tags else entry.kind
      lines.append(f"{tags}：{entry.text}")
    return lines

  def _compile_knowledge(self, entries: list[ExternalKnowledgeEntry]) -> list[str]:
    lines: list[str] = []
    for entry in entries[:self._limits.max_knowledge_items]:
      head = entry.topic or entry.category
      stance = str(entry.streamer_stance or "").strip()
      if head and entry.summary:
        line = f"{head}：{entry.summary}"
      elif entry.summary:
        line = entry.summary
      else:
        continue
      if stance:
        line = f"{line}\n主播立场：{stance}"
      lines.append(line)
    return lines
