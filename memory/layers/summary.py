"""
summary 层 — 定时总结的中长期记忆
每隔固定时间自动汇总浅层记忆，通过 significance 衰减实现遗忘
"""

import math
import uuid
from datetime import datetime
from typing import Optional

from ..config import SummaryConfig
from ..significance import (
  boost_significance,
  decay_significance,
  initial_significance,
)
from ..store import VectorStore
from ..archive import MemoryArchive
from .base import MemoryEntry


class SummaryLayer:
  """
  summary 记忆层

  定期汇总浅层记忆生成中长期记忆。
  使用 Chroma 向量存储 + significance 衰减 + 定期清理。
  """

  def __init__(
    self,
    vector_store: VectorStore,
    archive: MemoryArchive,
    config: Optional[SummaryConfig] = None,
  ):
    """
    初始化 summary 层

    Args:
      vector_store: 向量存储实例（collection: "summary"）
      archive: 归档器实例
      config: 层配置
    """
    self._store = vector_store
    self._archive = archive
    self._config = config or SummaryConfig()
    self.session_id: Optional[str] = None

  def add(self, content: str) -> str:
    """
    添加一条总结记忆

    Args:
      content: 总结内容

    Returns:
      记忆 ID
    """
    memory_id = str(uuid.uuid4())

    metadata = {
      "id": memory_id,
      "layer": "summary",
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "significance": initial_significance(),
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
    top_k: int = 2,
  ) -> list[MemoryEntry]:
    """
    RAG 检索并更新 significance

    Args:
      query: 查询文本
      top_k: 返回的最大结果数

    Returns:
      MemoryEntry 列表
    """
    if self._store.count() == 0:
      return []

    results = self._store.search(query=query, top_k=top_k)
    retrieved_ids = set()
    entries = []

    for doc, score in results:
      retrieved_ids.add(doc.metadata.get("id", ""))
      old_sig = doc.metadata.get("significance", initial_significance())
      new_sig = boost_significance(old_sig)

      ts_str = doc.metadata.get("timestamp", "")
      try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
      except (ValueError, TypeError):
        ts = datetime.now()

      entries.append(MemoryEntry(
        id=doc.metadata.get("id", ""),
        content=doc.page_content,
        layer="summary",
        timestamp=ts,
        significance=new_sig,
        score=score,
        metadata=doc.metadata,
      ))

    # 衰减未取用的记忆
    self._decay_unretrieved(retrieved_ids)

    return entries

  def _decay_unretrieved(self, retrieved_ids: set[str]) -> None:
    """衰减未取用记忆的 significance"""
    all_data = self._store.get_all()

    for i, doc_id in enumerate(all_data["ids"]):
      meta = all_data["metadatas"][i]
      old_sig = meta.get("significance", initial_significance())

      if doc_id in retrieved_ids:
        new_sig = boost_significance(old_sig)
      else:
        new_sig = decay_significance(old_sig, self._config.decay_coefficient)

      self._store.update_metadata(doc_id, {**meta, "significance": new_sig})

  def cleanup(self) -> int:
    """
    定期清理：删除 significance 最低的记忆

    删除比例由 config.cleanup_ratio 决定，向下取整。

    Returns:
      删除的记忆数量
    """
    total = self._store.count()
    if total == 0:
      return 0

    num_to_delete = int(math.floor(total * self._config.cleanup_ratio))
    if num_to_delete == 0:
      return 0

    # 获取所有数据，按 significance 排序找到最低的
    all_data = self._store.get_all()
    scored = []
    for i, doc_id in enumerate(all_data["ids"]):
      meta = all_data["metadatas"][i]
      sig = meta.get("significance", initial_significance())
      content = all_data["documents"][i] if all_data["documents"] else ""
      scored.append((doc_id, sig, content, meta))

    scored.sort(key=lambda x: x[1])  # 按 significance 升序
    to_delete = scored[:num_to_delete]

    # 删除并归档
    ids_to_delete = [item[0] for item in to_delete]
    memories_to_archive = [
      {
        "id": item[0],
        "content": item[2],
        "layer": "summary",
        "metadata": item[3],
      }
      for item in to_delete
    ]

    self._store.delete(ids_to_delete)
    self._archive.archive_batch(memories_to_archive)

    return num_to_delete

  def count(self) -> int:
    """获取当前记忆数量"""
    return self._store.count()

  def clear(self) -> None:
    """清空所有记忆"""
    self._store.clear()
