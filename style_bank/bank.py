"""
风格参考库（StyleBank）
从 persona 目录的 corpus.jsonl 加载语料到 Chroma 向量库，
按当前情境语义检索最相关的几条示例，格式化为 prompt 片段。
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from memory.store import VectorStore

logger = logging.getLogger(__name__)

SITUATION_MAP = {
  "react_comment": "回应弹幕",
  "react_scene": "描述画面",
  "proactive": "主动发言",
  "comeback": "回击质疑",
  "any": None,
}


class StyleBank:
  """
  风格参考库

  初始化时将 corpus.jsonl 全量灌入 Chroma（EphemeralClient），
  运行时按情境做语义检索，返回格式化文本供 _build_extra_context 注入。
  """

  def __init__(
    self,
    persona_dir: str | Path,
    embeddings: Optional[HuggingFaceEmbeddings] = None,
  ):
    persona_dir = Path(persona_dir)
    bank_dir = persona_dir / "style_bank"

    meta_path = bank_dir / "meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
      self._meta = json.load(f)

    # corpus_path 支持相对路径（相对于项目根目录）或绝对路径
    custom_corpus = self._meta.get("corpus_path")
    if custom_corpus:
      project_root = Path(__file__).resolve().parent.parent
      corpus_path = project_root / custom_corpus
    else:
      corpus_path = bank_dir / "corpus.jsonl"

    self._retrieval_count: int = self._meta.get("retrieval_count", 3)
    self._injection_probability: float = self._meta.get("injection_probability", 1.0)
    self._header: str = self._meta.get("injection_header", "【风格参考】")
    self._categories: dict[str, str] = self._meta.get("categories", {})
    self._next_trigger: Optional[bool] = None

    collection_name = f"style_bank_{persona_dir.name}"
    self._store = VectorStore(
      collection_name=collection_name,
      embeddings=embeddings,
    )

    self._load_corpus(corpus_path)

  def _load_corpus(self, corpus_path: Path) -> None:
    """批量加载 JSONL 语料到向量库"""
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []

    with open(corpus_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        item = json.loads(line)
        ids.append(item["id"])
        texts.append(item["text"])
        metas.append({
          "category": item.get("category", ""),
          "situation": item.get("situation", "any"),
          "score": item.get("score", 3),
        })

    if not ids:
      logger.warning("StyleBank 语料为空: %s", corpus_path)
      return

    batch_size = 500
    for i in range(0, len(ids), batch_size):
      self._store.add_batch(
        doc_ids=ids[i:i + batch_size],
        contents=texts[i:i + batch_size],
        metadatas=metas[i:i + batch_size],
      )

    logger.info("StyleBank 已加载 %d 条语料", len(ids))

  def pre_roll(self) -> bool:
    """预判本轮是否触发风格注入，缓存结果供 retrieve() 使用"""
    self._next_trigger = random.random() <= self._injection_probability
    return self._next_trigger

  def retrieve(
    self,
    query: str,
    situation: Optional[str] = None,
    top_k: Optional[int] = None,
  ) -> str:
    """
    按情境语义检索风格示例，返回格式化文本

    Args:
      query: 当前场景描述或弹幕内容
      situation: 情境标签（react_comment / react_scene / proactive / comeback）
      top_k: 覆盖默认检索数量

    Returns:
      格式化后的风格参考文本，无结果时返回空字符串
    """
    triggered = self._next_trigger if self._next_trigger is not None else (
      random.random() <= self._injection_probability
    )
    self._next_trigger = None
    if not triggered:
      return ""

    k = top_k or self._retrieval_count

    where = None
    if situation and situation != "any":
      where = {
        "$or": [
          {"situation": situation},
          {"situation": "any"},
        ]
      }

    results = self._store.search(query=query, top_k=k, where=where)

    if not results:
      return ""

    lines = [self._header]
    for i, (doc, _score) in enumerate(results, 1):
      cat = doc.metadata.get("category", "")
      cat_label = self._categories.get(cat, cat)
      lines.append(f"{i}. [{cat_label}] {doc.page_content}")

    return "\n".join(lines)

  def debug_state(self) -> dict:
    """调试快照"""
    return {
      "corpus_count": self._store.count(),
      "retrieval_count": self._retrieval_count,
      "categories": list(self._categories.keys()),
    }
