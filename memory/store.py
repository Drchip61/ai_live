"""
向量存储封装
为各记忆层提供统一的 Chroma 向量数据库操作接口
"""

import logging
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
    self._store.add_documents(documents=documents, ids=doc_ids)

  def search(
    self,
    query: str,
    top_k: int = 5,
    where: Optional[dict] = None,
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
    kwargs: dict = {"k": top_k}
    if where is not None:
      kwargs["filter"] = where

    return self._store.similarity_search_with_score(
      query=query,
      **kwargs,
    )

  def delete(self, doc_ids: list[str]) -> None:
    """删除指定文档"""
    if doc_ids:
      self._store.delete(ids=doc_ids)

  def update_metadata(self, doc_id: str, metadata: dict) -> None:
    """
    更新文档元数据

    Args:
      doc_id: 文档 ID
      metadata: 新的元数据（完整替换）
    """
    collection = self._store._collection
    collection.update(ids=[doc_id], metadatas=[metadata])

  def get_all(self) -> dict:
    """获取 collection 中所有文档数据"""
    return self._store._collection.get()

  def count(self) -> int:
    """获取文档总数"""
    return self._store._collection.count()

  def clear(self) -> None:
    """清空所有文档"""
    all_data = self._store._collection.get()
    if all_data["ids"]:
      self._store._collection.delete(ids=all_data["ids"])
