"""
viewer 层 — 按观众（user_id）索引的长期记忆
存储经小 LLM 筛选并改写为陈述性句子的观众记忆，支持按 user_id 精确召回和语义检索两种模式
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from ..config import ViewerConfig
from ..significance import (
  boost_significance,
  decay_significance,
  initial_significance,
)
from ..store import VectorStore
from ..archive import MemoryArchive
from .base import MemoryEntry

logger = logging.getLogger(__name__)


class ViewerMemoryLayer:
  """
  观众记忆层

  存储经小 LLM 筛选并改写的观众记忆（陈述性句子而非原始弹幕）。
  支持两种检索模式：
  1. 按 user_id 精确过滤：当某观众发弹幕时，召回该观众的历史记忆
  2. 按内容语义检索：不限用户，找"谁有过类似经历/说法"

  容量管理：全局 max_capacity + 每用户 max_per_user 双重限制。
  """

  def __init__(
    self,
    vector_store: VectorStore,
    archive: MemoryArchive,
    config: Optional[ViewerConfig] = None,
  ):
    self._store = vector_store
    self._archive = archive
    self._config = config or ViewerConfig()
    self.session_id: Optional[str] = None

  def add(
    self,
    user_id: str,
    nickname: str,
    content: str,
    ai_response_summary: str = "",
  ) -> str:
    """
    添加一条观众记忆

    Args:
      user_id: 观众 ID
      nickname: 观众昵称
      content: LLM 改写后的陈述性记忆
      ai_response_summary: AI 当时的回复摘要（可选）

    Returns:
      记忆 ID
    """
    if self._store.count() >= self._config.max_capacity:
      self._evict_least_significant()

    self._enforce_per_user_limit(user_id)

    memory_id = str(uuid.uuid4())
    metadata: dict = {
      "id": memory_id,
      "layer": "viewer",
      "user_id": user_id,
      "nickname": nickname,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "significance": initial_significance(),
    }
    if ai_response_summary:
      metadata["ai_response"] = ai_response_summary[:200]
    if self.session_id is not None:
      metadata["session_id"] = self.session_id

    self._store.add(
      doc_id=memory_id,
      content=content,
      metadata=metadata,
    )
    return memory_id

  def retrieve_by_user(
    self,
    user_id: str,
    query: str = "",
    top_k: int = 3,
  ) -> list[MemoryEntry]:
    """
    按 user_id 精确召回该观众的历史记忆

    有 query 时按语义相关性排序；无 query 时返回最近的记忆。
    命中的记忆 significance 提升。注意：不触发全局衰减，
    需由调用方在所有用户检索完成后统一调用 decay_unretrieved()。

    Args:
      user_id: 观众 ID
      query: 可选的语义查询（用当前弹幕内容检索最相关的历史）
      top_k: 返回条数

    Returns:
      MemoryEntry 列表
    """
    if self._store.count() == 0:
      return []

    search_query = query if query.strip() else user_id
    results = self._store.search(
      query=search_query,
      top_k=top_k,
      where={"user_id": user_id},
    )

    entries = []
    for doc, score in results:
      mem_id = doc.metadata.get("id", "")
      old_sig = doc.metadata.get("significance", initial_significance())
      new_sig = boost_significance(old_sig)
      self._store.update_metadata(mem_id, {**doc.metadata, "significance": new_sig})

      ts_str = doc.metadata.get("timestamp", "")
      try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
      except (ValueError, TypeError):
        ts = datetime.now()

      entries.append(MemoryEntry(
        id=mem_id,
        content=doc.page_content,
        layer="viewer",
        timestamp=ts,
        significance=new_sig,
        score=score,
        metadata=doc.metadata,
      ))

    return entries

  def retrieve_by_content(
    self,
    query: str,
    top_k: int = 3,
  ) -> list[MemoryEntry]:
    """
    纯语义检索（不限用户）

    Args:
      query: 查询文本
      top_k: 返回条数

    Returns:
      MemoryEntry 列表
    """
    if self._store.count() == 0 or not query.strip():
      return []

    results = self._store.search(query=query, top_k=top_k)
    entries = []
    for doc, score in results:
      ts_str = doc.metadata.get("timestamp", "")
      try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
      except (ValueError, TypeError):
        ts = datetime.now()

      entries.append(MemoryEntry(
        id=doc.metadata.get("id", ""),
        content=doc.page_content,
        layer="viewer",
        timestamp=ts,
        significance=doc.metadata.get("significance"),
        score=score,
        metadata=doc.metadata,
      ))

    return entries

  def decay_unretrieved(self, retrieved_ids: set[str]) -> None:
    """
    极慢衰减未命中记忆的 significance，仅用于区分淘汰优先级，不主动删除。

    应在一轮检索中所有 retrieve_by_user 调用完成后统一调用一次，
    避免多用户检索时重复遍历全量记忆。

    Args:
      retrieved_ids: 本轮所有被命中的记忆 ID 集合
    """
    all_data = self._store.get_all()
    for i, doc_id in enumerate(all_data["ids"]):
      if doc_id in retrieved_ids:
        continue
      meta = all_data["metadatas"][i]
      old_sig = meta.get("significance", initial_significance())
      new_sig = decay_significance(old_sig, self._config.decay_coefficient)
      self._store.update_metadata(doc_id, {**meta, "significance": new_sig})

  def _enforce_per_user_limit(self, user_id: str) -> None:
    """确保单个用户的记忆数不超过 max_per_user，超出时淘汰最旧的"""
    all_data = self._store.get_all()
    user_indices = [
      i for i, meta in enumerate(all_data["metadatas"])
      if meta.get("user_id") == user_id
    ]
    if len(user_indices) < self._config.max_per_user:
      return

    user_entries = [
      (all_data["ids"][i], all_data["metadatas"][i].get("significance", 1.0))
      for i in user_indices
    ]
    user_entries.sort(key=lambda x: x[1])
    evict_id = user_entries[0][0]

    idx = all_data["ids"].index(evict_id)
    content = all_data["documents"][idx] if all_data["documents"] else ""
    self._archive.archive_batch([{
      "id": evict_id,
      "content": content,
      "layer": "viewer",
      "metadata": all_data["metadatas"][idx],
    }])
    self._store.delete([evict_id])

  def _evict_least_significant(self) -> None:
    """淘汰全局 significance 最低的一条记忆"""
    all_data = self._store.get_all()
    if not all_data["ids"]:
      return
    min_idx = min(
      range(len(all_data["ids"])),
      key=lambda i: all_data["metadatas"][i].get("significance", 1.0),
    )
    evicted_id = all_data["ids"][min_idx]
    content = all_data["documents"][min_idx] if all_data["documents"] else ""
    self._archive.archive_batch([{
      "id": evicted_id,
      "content": content,
      "layer": "viewer",
      "metadata": all_data["metadatas"][min_idx],
    }])
    self._store.delete([evicted_id])

  def count(self) -> int:
    return self._store.count()

  def count_users(self) -> int:
    """获取有记忆的观众数量"""
    all_data = self._store.get_all()
    user_ids = {
      meta.get("user_id") for meta in all_data.get("metadatas", [])
      if meta.get("user_id")
    }
    return len(user_ids)

  def clear(self) -> None:
    self._store.clear()

  def debug_state(self) -> dict:
    """调试快照"""
    all_data = self._store.get_all()
    user_counts: dict[str, int] = {}
    for meta in all_data.get("metadatas", []):
      uid = meta.get("user_id", "unknown")
      user_counts[uid] = user_counts.get(uid, 0) + 1

    return {
      "total_memories": self._store.count(),
      "unique_users": len(user_counts),
      "per_user_counts": user_counts,
    }
