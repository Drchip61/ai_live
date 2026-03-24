"""
记忆系统全局配置

当前主链只保留：
- Active 短期记忆
- Structured context 存储与检索
"""

import os
from dataclasses import dataclass, field
from typing import Optional


def _default_persist_directory() -> Optional[str]:
  """允许用环境变量临时切换记忆库目录，便于离线修复验证。"""
  return os.getenv("MEMORY_PERSIST_DIRECTORY", "data/memory_store")


@dataclass(frozen=True)
class ActiveConfig:
  """Active 层配置"""
  capacity: int = 8


@dataclass(frozen=True)
class SummaryConfig:
  """定时汇总配置"""
  interval_seconds: float = 60.0


@dataclass(frozen=True)
class EmbeddingConfig:
  """嵌入模型配置"""
  model_name: str = "BAAI/bge-small-zh-v1.5"
  persist_directory: Optional[str] = field(default_factory=_default_persist_directory)


@dataclass(frozen=True)
class StructuredContextConfig:
  """结构化 memory/context 存储配置"""
  enabled: bool = True
  directory_name: str = "structured"
  user_memory_filename: str = "user_memory.json"
  self_memory_filename: str = "self_memory.json"
  persona_spec_filename: str = "persona_spec.json"
  corpus_filename: str = "corpus_store.json"
  external_knowledge_filename: str = "external_knowledge.json"
  use_as_primary_context: bool = True
  collection_prefix: str = "structured_"
  max_viewers: int = 2
  user_fact_top_k: int = 4
  user_recent_state_top_k: int = 2
  user_topic_top_k: int = 3
  user_callback_top_k: int = 2
  user_open_thread_top_k: int = 2
  user_sensitive_top_k: int = 2
  self_said_top_k: int = 3
  self_commitment_top_k: int = 2
  self_thread_top_k: int = 2
  persona_top_k: int = 4
  corpus_top_k: int = 3
  knowledge_top_k: int = 3
  semantic_max_distance: float = 1.5


@dataclass(frozen=True)
class MemoryConfig:
  """记忆系统总配置"""
  active: ActiveConfig = field(default_factory=ActiveConfig)
  summary: SummaryConfig = field(default_factory=SummaryConfig)
  embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
  structured: StructuredContextConfig = field(default_factory=StructuredContextConfig)
