"""
已删除记忆的归档管理
将被遗忘的记忆持久化到 personas/{角色}/archived_memories/ 下
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryArchive:
  """
  记忆归档器

  将已删除的记忆写入 JSON 文件，不再被取用。
  归档文件路径：personas/{persona}/archived_memories/archive.json
  """

  def __init__(
    self,
    persona: str,
    personas_dir: Optional[Path] = None,
    enabled: bool = True,
  ):
    """
    初始化归档器

    Args:
      persona: 角色名称
      personas_dir: personas 根目录，默认为项目根目录下的 personas/
      enabled: 是否启用归档（关闭时所有操作为 no-op）
    """
    self._enabled = enabled

    if personas_dir is None:
      # 从 memory/ 向上找到项目根目录
      project_root = Path(__file__).parent.parent
      personas_dir = project_root / "personas"

    self._archive_dir = personas_dir / persona / "archived_memories"
    self._archive_file = self._archive_dir / "archive.json"

  def archive(
    self,
    memory_id: str,
    content: str,
    layer: str,
    metadata: dict,
  ) -> None:
    """
    归档一条已删除的记忆

    Args:
      memory_id: 记忆 ID
      content: 记忆内容
      layer: 来源层级
      metadata: 原始元数据
    """
    if not self._enabled:
      return

    entry = {
      "id": memory_id,
      "content": content,
      "layer": layer,
      "metadata": metadata,
      "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 读取现有归档
    entries = self._load()
    entries.append(entry)
    self._save(entries)

  def archive_batch(
    self,
    memories: list[dict],
  ) -> None:
    """
    批量归档已删除的记忆

    Args:
      memories: 记忆字典列表，每个须含 id, content, layer, metadata
    """
    if not self._enabled or not memories:
      return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries = self._load()

    for mem in memories:
      entries.append({
        "id": mem["id"],
        "content": mem["content"],
        "layer": mem["layer"],
        "metadata": mem["metadata"],
        "archived_at": now,
      })

    self._save(entries)

  def _load(self) -> list[dict]:
    """加载现有归档数据"""
    if not self._archive_file.exists():
      return []
    try:
      return json.loads(
        self._archive_file.read_text(encoding="utf-8")
      )
    except (json.JSONDecodeError, IOError) as e:
      logger.error("读取归档文件失败: %s", e)
      return []

  def _save(self, entries: list[dict]) -> None:
    """保存归档数据"""
    self._archive_dir.mkdir(parents=True, exist_ok=True)
    self._archive_file.write_text(
      json.dumps(entries, ensure_ascii=False, indent=2),
      encoding="utf-8",
    )
