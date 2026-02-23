"""
memory 模块
分层记忆系统：active / temporary / summary / static

提供基于 RAG 的长期记忆能力，独立于 langchain_wrapper 和 streaming_studio，
桥接两层之间的记忆读写需求。
"""

from .config import (
  MemoryConfig,
  ActiveConfig,
  TemporaryConfig,
  SummaryConfig,
  RetrievalConfig,
  EmbeddingConfig,
  STATIC_CATEGORY_PREFIXES,
)
from .significance import (
  decay_significance,
  boost_significance,
  initial_significance,
)
from .store import VectorStore
from .archive import MemoryArchive
from .layers import (
  MemoryEntry,
  ActiveLayer,
  TemporaryLayer,
  SummaryLayer,
  StaticLayer,
)
from .retriever import MemoryRetriever
from .manager import MemoryManager
from .formatter import format_active_memories, format_retrieved_memories

__all__ = [
  # 配置
  "MemoryConfig",
  "ActiveConfig",
  "TemporaryConfig",
  "SummaryConfig",
  "RetrievalConfig",
  "EmbeddingConfig",
  "STATIC_CATEGORY_PREFIXES",
  # significance
  "decay_significance",
  "boost_significance",
  "initial_significance",
  # 存储
  "VectorStore",
  "MemoryArchive",
  # 层级
  "MemoryEntry",
  "ActiveLayer",
  "TemporaryLayer",
  "SummaryLayer",
  "StaticLayer",
  # 检索
  "MemoryRetriever",
  # 管理器
  "MemoryManager",
  # 格式化
  "format_active_memories",
  "format_retrieved_memories",
]
