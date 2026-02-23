"""
static 层 — 预设记忆
从 personas/{角色}/static_memories/ 加载，永不遗忘
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import STATIC_CATEGORY_PREFIXES, STATIC_CATEGORY_DEFAULT_PREFIX
from ..store import VectorStore
from .base import MemoryEntry

logger = logging.getLogger(__name__)


class StaticLayer:
  """
  static 记忆层

  从角色目录下的 static_memories/ 文件夹读取 JSON 文件，
  初始化时一次性写入向量存储。永不遗忘。

  JSON 文件格式：
    [
      {"content": "我叫花凛，今年17岁", "category": "identity"},
      {"content": "我认识一个叫小明的观众", "category": "relationship"},
      ...
    ]

  category 对应的检索前缀定义在 config.STATIC_CATEGORY_PREFIXES 中。
  """

  def __init__(
    self,
    vector_store: VectorStore,
    persona: str,
    personas_dir: Optional[Path] = None,
  ):
    """
    初始化 static 层

    Args:
      vector_store: 向量存储实例（collection: "static"）
      persona: 角色名称
      personas_dir: personas 根目录
    """
    self._store = vector_store
    self._persona = persona

    if personas_dir is None:
      project_root = Path(__file__).parent.parent.parent
      personas_dir = project_root / "personas"

    self._static_dir = personas_dir / persona / "static_memories"
    self._loaded = False

  def load(self) -> int:
    """
    从 JSON 文件加载静态记忆到向量存储

    如果已加载（向量存储非空），跳过加载。

    Returns:
      加载的记忆数量
    """
    if self._store.count() > 0:
      self._loaded = True
      return 0

    if not self._static_dir.exists():
      logger.info("静态记忆目录不存在: %s，跳过加载", self._static_dir)
      self._loaded = True
      return 0

    total = 0
    for json_file in sorted(self._static_dir.glob("*.json")):
      count = self._load_file(json_file)
      total += count

    self._loaded = True
    logger.info("加载了 %d 条静态记忆 (角色: %s)", total, self._persona)
    return total

  def _load_file(self, path: Path) -> int:
    """加载单个 JSON 文件"""
    try:
      data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError) as e:
      logger.error("读取静态记忆文件失败 %s: %s", path, e)
      return 0

    if not isinstance(data, list):
      logger.error("静态记忆文件格式错误（需要数组）: %s", path)
      return 0

    count = 0
    for item in data:
      content = item.get("content", "").strip()
      category = item.get("category", "").strip()
      if not content:
        continue

      memory_id = str(uuid.uuid4())
      self._store.add(
        doc_id=memory_id,
        content=content,
        metadata={
          "id": memory_id,
          "layer": "static",
          "category": category,
          "source_file": path.name,
          "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
      )
      count += 1

    return count

  def retrieve(
    self,
    query: str,
    top_k: int = 2,
  ) -> list[MemoryEntry]:
    """
    RAG 检索静态记忆

    Args:
      query: 查询文本
      top_k: 返回的最大结果数

    Returns:
      MemoryEntry 列表（content 已加上 category 前缀）
    """
    if not self._loaded:
      self.load()

    if self._store.count() == 0:
      return []

    results = self._store.search(query=query, top_k=top_k)
    entries = []

    for doc, score in results:
      category = doc.metadata.get("category", "")
      prefix = STATIC_CATEGORY_PREFIXES.get(category, STATIC_CATEGORY_DEFAULT_PREFIX)
      prefixed_content = f"{prefix}{doc.page_content}"

      ts_str = doc.metadata.get("timestamp", "")
      try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
      except (ValueError, TypeError):
        ts = datetime.now()

      entries.append(MemoryEntry(
        id=doc.metadata.get("id", ""),
        content=prefixed_content,
        layer="static",
        timestamp=ts,
        score=score,
        metadata=doc.metadata,
      ))

    return entries

  def count(self) -> int:
    """获取静态记忆数量"""
    return self._store.count()
