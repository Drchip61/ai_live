"""
stance 层 — AI 自身立场/观点的长期记忆
存储 AI 在对话中表达过的主观观点、偏好和立场，
通过极慢的 significance 衰减实现近乎永久的记忆保持
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from ..config import StanceConfig
from ..significance import (
  boost_significance,
  decay_significance,
  initial_significance,
)
from ..store import VectorStore
from ..archive import MemoryArchive
from .base import MemoryEntry

logger = logging.getLogger(__name__)

STANCE_INITIAL_SIGNIFICANCE = 0.700


class StanceLayer:
  """
  立场记忆层

  使用 Chroma 向量存储，每条立场带 significance 评分。
  衰减极慢（默认 0.995），确保 AI 的观点长期保持一致。
  写入前检查已有立场是否存在同话题冲突，冲突时标记旧立场为"已演变"。
  """

  def __init__(
    self,
    vector_store: VectorStore,
    archive: MemoryArchive,
    config: Optional[StanceConfig] = None,
  ):
    self._store = vector_store
    self._archive = archive
    self._config = config or StanceConfig()
    self.session_id: Optional[str] = None

  def add(
    self,
    content: str,
    topic: str,
    response_excerpt: str = "",
  ) -> str:
    """
    添加一条立场记忆

    写入前自动检测同话题已有立场，存在冲突时标记旧立场为"已演变"。

    Args:
      content: 立场内容（第一人称描述，如"我觉得XXX比较好"）
      topic: 话题关键词（如"最强中单英雄"）
      response_excerpt: 产生此立场的回复原文节选

    Returns:
      新立场的 ID
    """
    previous_id = self._detect_and_supersede(content, topic)

    memory_id = str(uuid.uuid4())
    metadata: dict = {
      "id": memory_id,
      "layer": "stance",
      "topic": topic,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "significance": STANCE_INITIAL_SIGNIFICANCE,
      "response_excerpt": response_excerpt[:200],
    }
    if previous_id:
      metadata["previous_stance_id"] = previous_id
    if self.session_id is not None:
      metadata["session_id"] = self.session_id

    self._store.add(
      doc_id=memory_id,
      content=content,
      metadata=metadata,
    )
    logger.info("立场记忆 +新增: [%s] %s", topic, content[:60])
    return memory_id

  def _detect_and_supersede(self, content: str, topic: str) -> Optional[str]:
    """
    检测同话题已有立场，存在时标记为"已演变"

    使用 topic 文本做语义搜索，距离低于阈值视为同话题。

    Returns:
      被取代的旧立场 ID，无冲突则返回 None
    """
    if self._store.count() == 0:
      return None

    results = self._store.search(query=topic, top_k=1)
    if not results:
      return None

    doc, score = results[0]
    if score > self._config.conflict_search_threshold:
      return None

    old_id = doc.metadata.get("id", "")
    if doc.metadata.get("superseded_by"):
      return None

    self._store.update_metadata(old_id, {
      **doc.metadata,
      "superseded_by": "pending",
      "significance": decay_significance(
        doc.metadata.get("significance", STANCE_INITIAL_SIGNIFICANCE),
        0.5,
      ),
    })
    logger.info(
      "立场演变: [%s] 旧=%s → 新立场即将写入",
      topic, doc.page_content[:40],
    )
    return old_id

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
      MemoryEntry 列表（仅返回未被取代的活跃立场）
    """
    if self._store.count() == 0:
      return []

    results = self._store.search(query=query, top_k=top_k * 2)
    retrieved_ids = set()
    entries = []

    for doc, score in results:
      if doc.metadata.get("superseded_by"):
        continue

      mem_id = doc.metadata.get("id", "")
      retrieved_ids.add(mem_id)

      old_sig = doc.metadata.get("significance", STANCE_INITIAL_SIGNIFICANCE)
      new_sig = boost_significance(old_sig)

      ts_str = doc.metadata.get("timestamp", "")
      try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
      except (ValueError, TypeError):
        ts = datetime.now()

      entries.append(MemoryEntry(
        id=mem_id,
        content=doc.page_content,
        layer="stance",
        timestamp=ts,
        significance=new_sig,
        score=score,
        metadata=doc.metadata,
      ))

      if len(entries) >= top_k:
        break

    self._decay_and_cleanup(retrieved_ids)

    return entries

  def _decay_and_cleanup(self, retrieved_ids: set[str]) -> None:
    """衰减未取用立场的 significance，删除低于阈值的"""
    if self._store.count() < self._config.min_count_before_decay:
      return

    all_data = self._store.get_all()
    ids_to_delete = []
    memories_to_archive = []

    for i, doc_id in enumerate(all_data["ids"]):
      meta = all_data["metadatas"][i]

      if doc_id in retrieved_ids:
        old_sig = meta.get("significance", STANCE_INITIAL_SIGNIFICANCE)
        new_sig = boost_significance(old_sig)
        self._store.update_metadata(doc_id, {**meta, "significance": new_sig})
      else:
        old_sig = meta.get("significance", STANCE_INITIAL_SIGNIFICANCE)
        new_sig = decay_significance(old_sig, self._config.decay_coefficient)

        if new_sig < self._config.significance_threshold:
          ids_to_delete.append(doc_id)
          content = all_data["documents"][i] if all_data["documents"] else ""
          memories_to_archive.append({
            "id": doc_id,
            "content": content,
            "layer": "stance",
            "metadata": meta,
          })
        else:
          self._store.update_metadata(doc_id, {**meta, "significance": new_sig})

    if ids_to_delete:
      self._store.delete(ids_to_delete)
      self._archive.archive_batch(memories_to_archive)

  def count(self) -> int:
    return self._store.count()

  def count_active(self) -> int:
    """获取未被取代的活跃立场数量"""
    all_data = self._store.get_all()
    return sum(
      1 for meta in all_data.get("metadatas", [])
      if not meta.get("superseded_by")
    )

  def clear(self) -> None:
    self._store.clear()
