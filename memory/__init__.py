"""
memory 模块

当前公开能力以 active + structured context 为主。
"""

from .config import (
  MemoryConfig,
  ActiveConfig,
  EmbeddingConfig,
  StructuredContextConfig,
)
from .context_schema import (
  UserMemoryRecord,
  SelfMemoryRecord,
  PersonaSpecRecord,
  CorpusEntry,
  ExternalKnowledgeEntry,
  CompiledMemoryContext,
)
from .context_store import (
  UserMemoryStore,
  SelfMemoryStore,
  PersonaSpecStore,
  CorpusStore,
  ExternalKnowledgeStore,
)
from .compiler import (
  CompilerLimits,
  MemoryCompiler,
  ContextCompiler,
)
from .structured_retriever import StructuredMemoryRetriever
from .store import VectorStore
from .layers import (
  ActiveLayer,
)
from .manager import MemoryManager

__all__ = [
  # 配置
  "MemoryConfig",
  "ActiveConfig",
  "EmbeddingConfig",
  "StructuredContextConfig",
  # 结构化 schema
  "UserMemoryRecord",
  "SelfMemoryRecord",
  "PersonaSpecRecord",
  "CorpusEntry",
  "ExternalKnowledgeEntry",
  "CompiledMemoryContext",
  # 结构化存储
  "UserMemoryStore",
  "SelfMemoryStore",
  "PersonaSpecStore",
  "CorpusStore",
  "ExternalKnowledgeStore",
  # 编译器
  "CompilerLimits",
  "MemoryCompiler",
  "ContextCompiler",
  "StructuredMemoryRetriever",
  # 存储
  "VectorStore",
  # 层级
  "ActiveLayer",
  # 管理器
  "MemoryManager",
]
