"""
向量存储封装
为各记忆层提供统一的 Chroma 向量数据库操作接口
"""

import logging
import threading
import time
from typing import Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class VectorStore:
  """
  Chroma 向量存储封装

  每个记忆层持有独立的 collection，共享同一个嵌入模型实例。
  所有操作通过 threading.RLock 串行化，防止 asyncio.to_thread 检索
  与事件循环线程写入并发访问 HNSW 索引导致 'Error finding id'。
  """

  def __init__(
    self,
    collection_name: str,
    config: Optional[EmbeddingConfig] = None,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
  ):
    """
    初始化向量存储

    Args:
      collection_name: Chroma collection 名称（每层各自独立）
      config: 嵌入模型配置
      embeddings: 共享的嵌入模型实例（传入以复用，不传则新建）
    """
    config = config or EmbeddingConfig()

    if embeddings is not None:
      self._embeddings = embeddings
    else:
      self._embeddings = HuggingFaceEmbeddings(
        model_name=config.model_name,
      )

    chroma_kwargs: dict = {
      "collection_name": collection_name,
      "embedding_function": self._embeddings,
    }
    if config.persist_directory is not None:
      chroma_kwargs["persist_directory"] = config.persist_directory

    self._store = Chroma(**chroma_kwargs)
    self._lock = threading.RLock()
    self._needs_heal = False

  @property
  def embeddings(self) -> HuggingFaceEmbeddings:
    """获取嵌入模型实例（供其他层复用）"""
    return self._embeddings

  def add(
    self,
    doc_id: str,
    content: str,
    metadata: dict,
  ) -> None:
    """
    添加单条文档

    Args:
      doc_id: 文档 ID
      content: 文本内容
      metadata: 元数据字典
    """
    with self._lock:
      self._store.add_documents(
        documents=[Document(page_content=content, metadata=metadata)],
        ids=[doc_id],
      )

  def add_batch(
    self,
    doc_ids: list[str],
    contents: list[str],
    metadatas: list[dict],
  ) -> None:
    """
    批量添加文档

    Args:
      doc_ids: 文档 ID 列表
      contents: 文本内容列表
      metadatas: 元数据列表
    """
    documents = [
      Document(page_content=content, metadata=meta)
      for content, meta in zip(contents, metadatas)
    ]
    with self._lock:
      self._store.add_documents(documents=documents, ids=doc_ids)

  @staticmethod
  def _build_documents(contents: list[str], metadatas: list[dict]) -> list[Document]:
    return [
      Document(page_content=content, metadata=meta)
      for content, meta in zip(contents, metadatas)
    ]

  def _embed_contents(self, contents: list[str]) -> list[list[float]]:
    if not contents:
      return []
    return self._embeddings.embed_documents(contents)

  def embed_query(self, query: str) -> list[float]:
    normalized = str(query or "").strip()
    if not normalized:
      return []
    return self._embeddings.embed_query(normalized)

  @property
  def collection_name(self) -> str:
    return self._store._collection.name

  @staticmethod
  def _is_recoverable_index_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
      "nothing found on disk",
      "hnsw segment reader",
      "error finding id",
    ))

  def _replace_documents_locked(
    self,
    *,
    delete_ids: list[str],
    doc_ids: list[str],
    contents: list[str],
    metadatas: list[dict],
    embeddings: Optional[list[list[float]]] = None,
  ) -> None:
    if delete_ids:
      self._store._collection.delete(ids=delete_ids)
    if doc_ids:
      self._store._collection.upsert(
        ids=doc_ids,
        documents=contents,
        metadatas=metadatas,
        embeddings=embeddings or self._embed_contents(contents),
      )

  def replace_all(
    self,
    doc_ids: list[str],
    contents: list[str],
    metadatas: list[dict],
  ) -> None:
    """原子替换整个 collection。"""
    embeddings = self._embed_contents(contents) if doc_ids else []
    with self._lock:
      all_data = self._store._collection.get()
      delete_ids = all_data.get("ids") or []
      self._replace_documents_locked(
        delete_ids=delete_ids,
        doc_ids=doc_ids,
        contents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
      )

  def replace_where(
    self,
    where: dict,
    doc_ids: list[str],
    contents: list[str],
    metadatas: list[dict],
  ) -> int:
    """原子替换满足 where 的文档集合，返回删除数量。"""
    embeddings = self._embed_contents(contents) if doc_ids else []
    with self._lock:
      data = self._store._collection.get(where=where)
      delete_ids = data.get("ids") or []
      self._replace_documents_locked(
        delete_ids=delete_ids,
        doc_ids=doc_ids,
        contents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
      )
      return len(delete_ids)

  def upsert_batch(
    self,
    doc_ids: list[str],
    contents: list[str],
    metadatas: list[dict],
  ) -> None:
    """
    批量 upsert 文档（已存在则覆盖，不存在则新增）
    """
    if not doc_ids:
      return
    embeddings = self._embed_contents(contents)
    with self._lock:
      self._store._collection.upsert(
        ids=doc_ids,
        documents=contents,
        metadatas=metadatas,
        embeddings=embeddings,
      )

  def search(
    self,
    query: str,
    top_k: int = 5,
    where: Optional[dict] = None,
    trace_collector: Optional[list[dict]] = None,
  ) -> list[tuple[Document, float]]:
    """
    语义相似度检索

    Args:
      query: 查询文本
      top_k: 返回的最大结果数
      where: Chroma 过滤条件

    Returns:
      (Document, score) 元组列表
    """
    embed_started = time.monotonic()
    query_embedding = self.embed_query(query)
    embed_query_ms = (time.monotonic() - embed_started) * 1000
    if not query_embedding:
      if trace_collector is not None:
        trace_collector.append({
          "collection_name": self.collection_name,
          "embed_query_ms": round(embed_query_ms, 1),
          "chroma_query_ms": 0.0,
          "retry_count": 0,
          "self_heal_ms": 0.0,
          "result_count": 0,
        })
      return []
    return self.search_by_vector(
      query_embedding,
      top_k=top_k,
      where=where,
      trace_collector=trace_collector,
      embed_query_ms=embed_query_ms,
    )

  @staticmethod
  def _query_results_to_docs_and_scores(results: dict) -> list[tuple[Document, float]]:
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    converted: list[tuple[Document, float]] = []
    for idx, content in enumerate(documents):
      metadata = {}
      if idx < len(metadatas) and isinstance(metadatas[idx], dict):
        metadata = metadatas[idx]
      score = distances[idx] if idx < len(distances) else 0.0
      converted.append((
        Document(page_content=str(content or ""), metadata=metadata),
        float(score or 0.0),
      ))
    return converted

  def _search_by_vector_locked(
    self,
    query_embedding: list[float],
    top_k: int,
    where: Optional[dict],
  ) -> list[tuple[Document, float]]:
    kwargs = {
      "query_embeddings": [query_embedding],
      "n_results": top_k,
      "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
      kwargs["where"] = where
    results = self._store._collection.query(**kwargs)
    return self._query_results_to_docs_and_scores(results)

  def search_by_vector(
    self,
    query_embedding: list[float],
    top_k: int = 5,
    where: Optional[dict] = None,
    trace_collector: Optional[list[dict]] = None,
    embed_query_ms: float = 0.0,
  ) -> list[tuple[Document, float]]:
    """复用已生成的 query embedding 做检索。"""
    if not query_embedding:
      return []

    results: list[tuple[Document, float]] = []
    chroma_query_ms = 0.0
    retry_count = 0
    self_heal_ms = 0.0
    query_started = time.monotonic()
    try:
      with self._lock:
        results = self._search_by_vector_locked(query_embedding, top_k, where)
        chroma_query_ms += (time.monotonic() - query_started) * 1000
    except Exception as e:
      chroma_query_ms += max(0.0, (time.monotonic() - query_started) * 1000)
      if self._is_recoverable_index_error(e):
        logger.warning(
          "向量检索索引异常 (collection=%s): %s，跳过本次检索",
          self.collection_name, e,
        )
        self._needs_heal = True
      else:
        logger.error("向量检索失败 (collection=%s): %s", self.collection_name, e)
      results = []
    if trace_collector is not None:
      trace_collector.append({
        "collection_name": self.collection_name,
        "embed_query_ms": round(embed_query_ms, 1),
        "chroma_query_ms": round(chroma_query_ms, 1),
        "retry_count": retry_count,
        "self_heal_ms": round(self_heal_ms, 1),
        "result_count": len(results),
      })
    return results

  def delete(self, doc_ids: list[str]) -> None:
    """删除指定文档"""
    if doc_ids:
      with self._lock:
        self._store.delete(ids=doc_ids)

  def update_metadata(self, doc_id: str, metadata: dict) -> None:
    """
    更新文档元数据

    Args:
      doc_id: 文档 ID
      metadata: 新的元数据（完整替换）
    """
    with self._lock:
      collection = self._store._collection
      collection.update(ids=[doc_id], metadatas=[metadata])

  def update_metadata_batch(self, doc_ids: list[str], metadatas: list[dict]) -> None:
    """
    批量更新文档元数据（单次 Chroma 调用，避免逐条 update 的巨大开销）

    Args:
      doc_ids: 文档 ID 列表
      metadatas: 元数据列表（与 doc_ids 一一对应）
    """
    if not doc_ids:
      return
    with self._lock:
      self._store._collection.update(ids=doc_ids, metadatas=metadatas)

  def get_all(self) -> dict:
    """获取 collection 中所有文档数据"""
    with self._lock:
      return self._store._collection.get()

  def get(self, where: Optional[dict] = None) -> dict:
    """按过滤条件获取文档数据"""
    with self._lock:
      kwargs = {}
      if where is not None:
        kwargs["where"] = where
      return self._store._collection.get(**kwargs)

  def delete_where(self, where: dict) -> int:
    """按过滤条件删除文档，返回删除数量"""
    with self._lock:
      data = self._store._collection.get(where=where)
      ids = data.get("ids") or []
      if ids:
        self._store._collection.delete(ids=ids)
      return len(ids)

  def search_raw(
    self,
    query: str,
    top_k: int = 1,
  ) -> list[tuple[Document, float]]:
    """与 search() 相同但不捕获异常，用于索引健康检查"""
    with self._lock:
      return self._store.similarity_search_with_score(query=query, k=top_k)

  def ensure_healthy(self) -> bool:
    """
    探测 HNSW 索引健康，损坏时自动重建（数据零丢失）。

    原理：get_all() 走 Chroma 底层 SQLite（不经过 HNSW），
    能完整导出所有记忆。clear + add_batch 重建 HNSW 索引。

    Returns:
      True 表示索引健康或为空，False 表示执行了自愈重建
    """
    collection_name = self.collection_name
    try:
      with self._lock:
        n = self._store._collection.count()
        if n == 0:
          return True
        self._store.similarity_search_with_score(query="health_check", k=1)
      return True
    except Exception as e:
      logger.warning(
        "HNSW 索引损坏 (collection=%s): %s，开始自愈重建",
        collection_name, e,
      )
      print(f"[记忆] {collection_name} 索引损坏，自愈重建中（数据零丢失）...")
      ids: list[str] = []
      docs: list[str] = []
      metas: list[dict] = []
      with self._lock:
        all_data = self._store._collection.get()
        ids = all_data.get("ids") or []
        docs = all_data.get("documents") or []
        metas = all_data.get("metadatas") or []
      embeddings = self._embed_contents(docs) if ids else []
      with self._lock:
        self._replace_documents_locked(
          delete_ids=ids,
          doc_ids=ids,
          contents=docs,
          metadatas=metas,
          embeddings=embeddings,
        )
      logger.info(
        "索引重建完成 (collection=%s)，%d 条记忆已恢复",
        collection_name, len(ids),
      )
      print(f"[记忆] {collection_name} 索引重建完成，{len(ids)} 条记忆已恢复")
      return False

  def heal_if_needed(self) -> bool:
    """有待修复标记时执行自愈，供后台任务调用。返回是否执行了重建。"""
    if not self._needs_heal:
      return False
    self._needs_heal = False
    return not self.ensure_healthy()

  def count(self) -> int:
    """获取文档总数"""
    with self._lock:
      return self._store._collection.count()

  def clear(self) -> None:
    """清空所有文档"""
    self.replace_all([], [], [])
