"""
跨层记忆检索器
支持两种检索模式：per-layer quota / weighted merge
支持多查询（逐条弹幕 RAG + 去重）
"""

from typing import Optional, Union

from langchain_core.runnables import RunnableLambda

from .config import RetrievalConfig
from .formatter import format_active_memories, format_retrieved_memories
from .layers.base import MemoryEntry
from .layers.active import ActiveLayer
from .layers.temporary import TemporaryLayer
from .layers.summary import SummaryLayer
from .layers.static import StaticLayer


def _dedup_entries(entries: list[MemoryEntry], top_k: int) -> list[MemoryEntry]:
  """
  按 ID 去重，保留每条记忆的最高分，取 top_k

  Args:
    entries: 可能包含重复 ID 的记忆列表
    top_k: 最终保留数量

  Returns:
    去重后按 score 排序的 top_k 条记忆
  """
  best: dict[str, MemoryEntry] = {}
  for entry in entries:
    existing = best.get(entry.id)
    if existing is None or entry.score < existing.score:
      # Chroma score 越小越相似
      best[entry.id] = entry
  sorted_entries = sorted(best.values(), key=lambda e: e.score)
  return sorted_entries[:top_k]


class MemoryRetriever:
  """
  跨层记忆检索器

  协调 active / temporary / summary / static 四层的检索，
  按配置模式合并结果，统一排序后输出。

  支持两种模式（通过 config.retrieval.mode 切换）：
  - "quota": 每层分配固定取回数量
  - "weighted": 所有 RAG 层合并取回，按层级系数加权后重排

  支持多查询输入：逐条查询 + 按 ID 去重。
  """

  def __init__(
    self,
    active: ActiveLayer,
    temporary: TemporaryLayer,
    summary: SummaryLayer,
    static: StaticLayer,
    config: Optional[RetrievalConfig] = None,
  ):
    """
    初始化检索器

    Args:
      active: active 层实例
      temporary: temporary 层实例
      summary: summary 层实例
      static: static 层实例
      config: 检索配置
    """
    self._active = active
    self._temporary = temporary
    self._summary = summary
    self._static = static
    self._config = config or RetrievalConfig()
    self.session_id: Optional[str] = None

  def retrieve(self, query: Union[str, list[str]]) -> tuple[str, str]:
    """
    执行跨层检索

    Args:
      query: 查询文本，支持单条字符串或多条列表。
        多条时逐条检索 + 按 ID 去重，语义匹配更精准。

    Returns:
      (active_text, rag_text) 元组：
        active_text: active 层格式化文本（时序直接注入）
        rag_text: RAG 层格式化文本（temporary + summary + static）
    """
    # 标准化为列表
    queries = [query] if isinstance(query, str) else query
    queries = [q for q in queries if q.strip()]
    if not queries:
      queries = [""]

    # active 层：全量直接取出（不走 RAG）
    active_memories = self._active.get_all()
    active_entries = [
      MemoryEntry(
        id=m.id,
        content=m.content,
        layer="active",
        timestamp=m.timestamp,
      )
      for m in active_memories
    ]
    active_text = format_active_memories(active_entries)

    # RAG 层检索
    if self._config.mode == "weighted":
      rag_entries = self._retrieve_weighted(queries)
    else:
      rag_entries = self._retrieve_quota(queries)

    rag_text = format_retrieved_memories(
      rag_entries,
      current_session_id=self.session_id,
    )

    return active_text, rag_text

  def _retrieve_quota(self, queries: list[str]) -> list[MemoryEntry]:
    """
    per-layer quota 模式：每层逐条查询，去重后取 quota 数量

    Args:
      queries: 查询文本列表
    """
    entries = []

    if self._config.quota_temporary > 0:
      all_temp = []
      for q in queries:
        all_temp.extend(
          self._temporary.retrieve(q, top_k=self._config.quota_temporary)
        )
      entries.extend(_dedup_entries(all_temp, self._config.quota_temporary))

    if self._config.quota_summary > 0:
      all_sum = []
      for q in queries:
        all_sum.extend(
          self._summary.retrieve(q, top_k=self._config.quota_summary)
        )
      entries.extend(_dedup_entries(all_sum, self._config.quota_summary))

    if self._config.quota_static > 0:
      all_stat = []
      for q in queries:
        all_stat.extend(
          self._static.retrieve(q, top_k=self._config.quota_static)
        )
      entries.extend(_dedup_entries(all_stat, self._config.quota_static))

    return entries

  def _retrieve_weighted(self, queries: list[str]) -> list[MemoryEntry]:
    """
    weighted 模式：所有层合并取回，按层级系数加权后重排

    多取回 overfetch_multiplier 倍的结果，加权后取 top-k。

    Args:
      queries: 查询文本列表
    """
    total_quota = (
      self._config.quota_temporary
      + self._config.quota_summary
      + self._config.quota_static
    )
    overfetch = total_quota * self._config.weighted_overfetch_multiplier

    # 每层多取回
    per_layer = max(1, overfetch // 3)
    all_entries = []

    for q in queries:
      all_entries.extend(self._temporary.retrieve(q, top_k=per_layer))
      all_entries.extend(self._summary.retrieve(q, top_k=per_layer))
      all_entries.extend(self._static.retrieve(q, top_k=per_layer))

    # 按 ID 去重（保留最佳 score）
    best: dict[str, MemoryEntry] = {}
    for entry in all_entries:
      existing = best.get(entry.id)
      if existing is None or entry.score < existing.score:
        best[entry.id] = entry

    # 加权打分
    weight_map = {
      "temporary": self._config.weight_temporary,
      "summary": self._config.weight_summary,
      "static": self._config.weight_static,
    }

    weighted = []
    for entry in best.values():
      w = weight_map.get(entry.layer, 1.0)
      # Chroma 返回的 score 越小越相似，加权时取倒数使大的更好
      weighted_score = w / (1.0 + entry.score)
      weighted.append((entry, weighted_score))

    # 按加权分降序排列，取 top-k
    weighted.sort(key=lambda x: x[1], reverse=True)
    return [entry for entry, _ in weighted[:total_quota]]

  def as_runnable(self) -> RunnableLambda:
    """
    获取 LCEL 兼容的 Runnable

    输入: dict（须含 "input" 键）
    输出: dict，原始数据 + "active_memories" + "retrieved_memories"

    Returns:
      RunnableLambda
    """
    def _invoke(data: dict) -> dict:
      query = data.get("input", "")
      if not query:
        return {**data, "active_memories": "", "retrieved_memories": ""}

      active_text, rag_text = self.retrieve(query)
      return {
        **data,
        "active_memories": active_text,
        "retrieved_memories": rag_text,
      }

    return RunnableLambda(_invoke)
