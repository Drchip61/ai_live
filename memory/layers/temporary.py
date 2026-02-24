"""
temporary 层 — 进入 RAG 流程的短期记忆
从 active 层溢出的记忆进入此层，通过 significance 衰减实现遗忘
"""

import uuid
from datetime import datetime
from typing import Optional

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
  ):
    """
    初始化 temporary 层

    Args:
      vector_store: 向量存储实例（collection: "temporary"）
      archive: 归档器实例
      config: 层配置
    """
    self._store = vector_store
    self._archive = archive
    self._config = config or TemporaryConfig()
    self.session_id: Optional[str] = None

  def add(self, content: str, timestamp: Optional[datetime] = None, response: str = "") -> str:
    """
    添加一条记忆（通常来自 active 层溢出）

    Args:
      content: 记忆内容
      timestamp: 原始时间戳（来自 active 层），默认为当前时间
      response: 主播当时的回复原文

    Returns:
      记忆 ID
    """
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
    retrieved_ids = set()
    entries = []

    for doc, score in results:
      mem_id = doc.metadata.get("id", "")
      retrieved_ids.add(mem_id)

      old_sig = doc.metadata.get("significance", initial_significance())
      new_sig = boost_significance(old_sig)

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
    self._decay_and_cleanup(retrieved_ids)

    return entries

  def _decay_and_cleanup(self, retrieved_ids: set[str]) -> None:
    """
    衰减未取用记忆的 significance，删除低于阈值的记忆

    Args:
      retrieved_ids: 本次被取用的记忆 ID 集合
    """
    all_data = self._store.get_all()
    ids_to_delete = []
    memories_to_archive = []

    for i, doc_id in enumerate(all_data["ids"]):
      if doc_id in retrieved_ids:
        # 被取用的：提升（已在 retrieve 中处理逻辑，此处更新存储）
        meta = all_data["metadatas"][i]
        old_sig = meta.get("significance", initial_significance())
        new_sig = boost_significance(old_sig)
        self._store.update_metadata(doc_id, {**meta, "significance": new_sig})
      else:
        # 未取用的：衰减
        meta = all_data["metadatas"][i]
        old_sig = meta.get("significance", initial_significance())
        new_sig = decay_significance(old_sig, self._config.decay_coefficient)

        if new_sig < self._config.significance_threshold:
          # 低于阈值，标记为待删除
          ids_to_delete.append(doc_id)
          content = all_data["documents"][i] if all_data["documents"] else ""
          memories_to_archive.append({
            "id": doc_id,
            "content": content,
            "layer": "temporary",
            "metadata": meta,
          })
        else:
          self._store.update_metadata(doc_id, {**meta, "significance": new_sig})

    # 删除并归档
    if ids_to_delete:
      self._store.delete(ids_to_delete)
      self._archive.archive_batch(memories_to_archive)

  def count(self) -> int:
    """获取当前记忆数量"""
    return self._store.count()

  def clear(self) -> None:
    """清空所有记忆"""
    self._store.clear()
