"""
记忆层级基类和通用数据结构
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class MemoryEntry:
  """
  统一的记忆条目

  跨层级的通用数据结构，用于检索结果的统一表示。
  """
  id: str
  content: str
  layer: str                          # "active" / "temporary" / "summary" / "static"
  timestamp: datetime
  significance: Optional[float] = None  # 仅 temporary / summary 层有值
  score: float = 0.0                  # RAG 相似度分数
  metadata: Optional[dict] = None     # 层级特有的元数据
