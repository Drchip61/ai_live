"""
跨层记忆检索器
支持两种检索模式：per-layer quota / weighted merge
支持多查询（逐条弹幕 RAG + 去重）
支持按 viewer_ids 召回观众历史记忆
"""

import logging
from typing import Optional, Union

from langchain_core.runnables import RunnableLambda

logger = logging.getLogger(__name__)

from .config import RetrievalConfig
from .formatter import (
  format_active_memories,
  format_retrieved_memories,
  format_viewer_memories,
)
from .layers.base import MemoryEntry
from .layers.active import ActiveLayer
from .layers.temporary import TemporaryLayer
from .layers.summary import SummaryLayer
from .layers.static import StaticLayer
from .layers.stance import StanceLayer
from .layers.viewer import ViewerMemoryLayer


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
    stance: Optional[StanceLayer] = None,
    viewer: Optional[ViewerMemoryLayer] = None,
    config: Optional[RetrievalConfig] = None,
  ):
    """
    初始化检索器

    Args:
      active: active 层实例
      temporary: temporary 层实例
      summary: summary 层实例
      static: static 层实例
      stance: stance 层实例（立场记忆，可选）
      viewer: viewer 层实例（观众记忆，可选）
      config: 检索配置
    """
    self._active = active
    self._temporary = temporary
    self._summary = summary
    self._static = static
    self._stance = stance
    self._viewer = viewer
    self._config = config or RetrievalConfig()
    self.session_id: Optional[str] = None

  def retrieve(
    self,
    query: Union[str, list[str]],
    viewer_ids: Optional[list[str]] = None,
  ) -> tuple[str, str, str]:
    """
    执行跨层检索

    Args:
      query: 查询文本，支持单条字符串或多条列表。
        多条时逐条检索 + 按 ID 去重，语义匹配更精准。
      viewer_ids: 当前弹幕中出现的观众 user_id 列表（用于召回观众历史记忆）

    Returns:
      (active_text, rag_text, viewer_text) 三元组：
        active_text: active 层格式化文本（时序直接注入）
        rag_text: RAG 层格式化文本（temporary + summary + static + stance）
        viewer_text: 观众记忆格式化文本（按 user_id 召回的历史发言）
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
        metadata={"response": m.response},
      )
      for m in active_memories
    ]
    active_text = format_active_memories(
      active_entries,
      include_response=self._config.include_response_in_active,
      response_max_length=self._config.response_display_max_length,
    )

    # RAG 层检索
    if self._config.mode == "weighted":
      rag_entries = self._retrieve_weighted(queries)
    else:
      rag_entries = self._retrieve_quota(queries)

    rag_text = format_retrieved_memories(
      rag_entries,
      current_session_id=self.session_id,
      include_temp_response=self._config.include_response_in_temporary,
      response_max_length=self._config.response_display_max_length,
    )

    # 观众记忆检索
    viewer_text = ""
    if self._viewer is not None and viewer_ids:
      viewer_entries = self._retrieve_viewer(queries, viewer_ids)
      viewer_text = format_viewer_memories(
        viewer_entries,
        current_session_id=self.session_id,
      )

    return active_text, rag_text, viewer_text

  def retrieve_active_only(self) -> tuple[str, str, str]:
    """
    仅返回 Active 层记忆，不触发任何 RAG 检索和 significance 衰减

    Returns:
      (active_text, "", "") 三元组
    """
    active_memories = self._active.get_all()
    active_entries = [
      MemoryEntry(
        id=m.id,
        content=m.content,
        layer="active",
        timestamp=m.timestamp,
        metadata={"response": m.response},
      )
      for m in active_memories
    ]
    active_text = format_active_memories(
      active_entries,
      include_response=self._config.include_response_in_active,
      response_max_length=self._config.response_display_max_length,
    )
    return active_text, "", ""

  def _retrieve_quota(self, queries: list[str]) -> list[MemoryEntry]:
    """
    per-layer quota 模式：每层逐条查询，去重后取 quota 数量

    Args:
      queries: 查询文本列表
    """
    entries = []

    layers = [
      ("temporary", self._temporary, self._config.quota_temporary),
      ("summary", self._summary, self._config.quota_summary),
      ("static", self._static, self._config.quota_static),
    ]
    if self._stance is not None:
      layers.append(("stance", self._stance, self._config.quota_stance))

    for layer_name, layer, quota in layers:
      if quota <= 0:
        continue
      try:
        all_results = []
        for q in queries:
          all_results.extend(layer.retrieve(q, top_k=quota))
        entries.extend(_dedup_entries(all_results, quota))
      except Exception as e:
        logger.error("记忆层 %s 检索失败，跳过: %s", layer_name, e)

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
      + (self._config.quota_stance if self._stance is not None else 0)
    )
    layer_count = 3 + (1 if self._stance is not None else 0)
    overfetch = total_quota * self._config.weighted_overfetch_multiplier

    per_layer = max(1, overfetch // layer_count)
    all_entries = []

    layers = [
      ("temporary", self._temporary),
      ("summary", self._summary),
      ("static", self._static),
    ]
    if self._stance is not None:
      layers.append(("stance", self._stance))

    for q in queries:
      for layer_name, layer in layers:
        try:
          all_entries.extend(layer.retrieve(q, top_k=per_layer))
        except Exception as e:
          logger.error("记忆层 %s 检索失败，跳过: %s", layer_name, e)

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
      "stance": self._config.weight_stance,
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

  def _retrieve_viewer(
    self,
    queries: list[str],
    viewer_ids: list[str],
  ) -> list[MemoryEntry]:
    """
    按 viewer_ids 召回观众历史记忆

    对每个 user_id，用弹幕内容做语义检索（带 user_id 过滤），
    多个用户的结果合并后按 ID 去重。
    所有用户检索完成后统一执行一次全局衰减。

    Args:
      queries: 查询文本列表（弹幕内容）
      viewer_ids: 观众 user_id 列表
    """
    if self._viewer is None:
      return []

    all_entries: list[MemoryEntry] = []
    seen_ids: set[str] = set()
    query_text = " ".join(queries) if queries else ""

    for uid in viewer_ids:
      entries = self._viewer.retrieve_by_user(
        user_id=uid,
        query=query_text,
        top_k=3,
      )
      for entry in entries:
        if entry.id not in seen_ids:
          seen_ids.add(entry.id)
          all_entries.append(entry)

    if seen_ids:
      self._viewer.decay_unretrieved(seen_ids)

    return all_entries

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

      active_text, rag_text, _viewer_text = self.retrieve(query)
      return {
        **data,
        "active_memories": active_text,
        "retrieved_memories": rag_text,
      }

    return RunnableLambda(_invoke)
