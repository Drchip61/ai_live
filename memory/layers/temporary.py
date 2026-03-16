"""
temporary 层 — 进入 RAG 流程的短期记忆
从 active 层溢出的记忆进入此层，通过 significance 衰减实现遗忘
"""

import uuid
from datetime import datetime
from typing import Optional

from collections import deque
from typing import Callable

from ..config import TemporaryConfig
from ..significance import (
  boost_significance,
  decay_significance,
  initial_significance,
)
from ..store import VectorStore
from ..archive import MemoryArchive
from .base import MemoryEntry


class TemporaryLayer:
  """
  temporary 记忆层

  使用 Chroma 向量存储，每条记忆带 significance 评分。
  检索后更新评分：被取用的提升，未取用的衰减。
  低于阈值的记忆被删除并归档。
  """

  def __init__(
    self,
    vector_store: VectorStore,
    archive: MemoryArchive,
    config: Optional[TemporaryConfig] = None,
    on_fade: Optional[Callable[[str], None]] = None,
  ):
    """
    初始化 temporary 层

    Args:
      vector_store: 向量存储实例（collection: "temporary"）
      archive: 归档器实例
      config: 层配置
      on_fade: 记忆即将遗忘时的回调，参数为记忆内容文本
    """
    self._store = vector_store
    self._archive = archive
    self._config = config or TemporaryConfig()
    self._on_fade = on_fade
    self.session_id: Optional[str] = None

  def add(self, content: str, timestamp: Optional[datetime] = None, response: str = "") -> str:
    """
    添加一条记忆（通常来自 active 层溢出）

    容量满时自动淘汰 significance 最低的记忆。

    Args:
      content: 记忆内容
      timestamp: 原始时间戳（来自 active 层），默认为当前时间
      response: 主播当时的回复原文

    Returns:
      记忆 ID
    """
    if self._store.count() >= self._config.max_capacity:
      self._evict_least_significant()

    memory_id = str(uuid.uuid4())
    ts = timestamp or datetime.now()

    metadata = {
      "id": memory_id,
      "layer": "temporary",
      "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
      "significance": initial_significance(),
      "response": response[:200],
    }
    if self.session_id is not None:
      metadata["session_id"] = self.session_id

    self._store.add(
      doc_id=memory_id,
      content=content,
      metadata=metadata,
    )
    return memory_id

  def retrieve(
    self,
    query: str,
    top_k: int = 3,
  ) -> list[MemoryEntry]:
    """
    RAG 检索并更新 significance

    被取用的记忆 significance 提升，所有未取用的衰减。
    低于阈值的记忆被删除并归档。

    Args:
      query: 查询文本
      top_k: 返回的最大结果数

    Returns:
      MemoryEntry 列表
    """
    if self._store.count() == 0:
      return []

    # 检索
    results = self._store.search(query=query, top_k=top_k)
    # retrieved_boosts: 被取用记忆的 ID → boost 后的 significance
    retrieved_boosts: dict[str, float] = {}
    entries = []

    for doc, score in results:
      mem_id = doc.metadata.get("id", "")
      old_sig = doc.metadata.get("significance", initial_significance())
      new_sig = boost_significance(old_sig)
      retrieved_boosts[mem_id] = new_sig

      ts_str = doc.metadata.get("timestamp", "")
      try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
      except (ValueError, TypeError):
        ts = datetime.now()

      entries.append(MemoryEntry(
        id=mem_id,
        content=doc.page_content,
        layer="temporary",
        timestamp=ts,
        significance=new_sig,
        score=score,
        metadata=doc.metadata,
      ))

    # 衰减所有未取用的记忆 + 清理低 significance 记忆
    self._decay_and_cleanup(retrieved_boosts)

    return entries

  def _evict_least_significant(self) -> None:
    """淘汰 significance 最低的一条记忆"""
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
      "layer": "temporary",
      "metadata": all_data["metadatas"][min_idx],
    }])
    self._store.delete([evicted_id])

  def _decay_and_cleanup(self, retrieved_boosts: dict[str, float]) -> None:
    """
    写回被取用记忆的 boosted significance，衰减未取用记忆，删除低于阈值的记忆

    Args:
      retrieved_boosts: 被取用的记忆 ID → boost 后的 significance 值
    """
    all_data = self._store.get_all()
    ids_to_delete = []
    memories_to_archive = []
    update_ids: list[str] = []
    update_metas: list[dict] = []

    for i, doc_id in enumerate(all_data["ids"]):
      meta = all_data["metadatas"][i]
      if doc_id in retrieved_boosts:
        update_ids.append(doc_id)
        update_metas.append({**meta, "significance": retrieved_boosts[doc_id]})
      else:
        old_sig = meta.get("significance", initial_significance())
        new_sig = decay_significance(old_sig, self._config.decay_coefficient)

        if new_sig < self._config.significance_threshold:
          ids_to_delete.append(doc_id)
          content = all_data["documents"][i] if all_data["documents"] else ""
          memories_to_archive.append({
            "id": doc_id,
            "content": content,
            "layer": "temporary",
            "metadata": meta,
          })
        else:
          update_ids.append(doc_id)
          update_metas.append({**meta, "significance": new_sig})

    if update_ids:
      self._store.update_metadata_batch(update_ids, update_metas)

    if ids_to_delete:
      if self._on_fade:
        for mem in memories_to_archive:
          content = mem.get("content", "")
          if content:
            self._on_fade(content)
      self._store.delete(ids_to_delete)
      self._archive.archive_batch(memories_to_archive)

  def count(self) -> int:
    """获取当前记忆数量"""
    return self._store.count()

  def clear(self) -> None:
    """清空所有记忆"""
    self._store.clear()
