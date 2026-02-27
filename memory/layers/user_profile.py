"""
用户画像层
结构化存储用户的稳定个人信息（偏好、习惯、职业等）
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CATEGORIES = ("likes", "dislikes", "habits", "occupation", "pets", "catchphrases", "other")


class UserProfileLayer:
  """
  用户画像存储层

  以 JSON 结构存储用户信息，每条数据包含 value / source / confidence。
  支持冲突覆盖（新信息替换旧信息，旧信息记入变更历史）。
  """

  def __init__(self, persist_path: Optional[Path] = None) -> None:
    self._data: dict[str, Any] = {cat: {} for cat in _CATEGORIES}
    self._change_history: list[dict] = []
    self._persist_path = persist_path
    if persist_path and persist_path.exists():
      self._load_from_disk()

  def get(self, category: str, key: Optional[str] = None) -> Any:
    bucket = self._data.get(category, {})
    if key is None:
      return bucket
    if isinstance(bucket, dict) and not isinstance(bucket.get("value"), str):
      return bucket.get(key)
    return bucket

  def set(
    self,
    category: str,
    key: str,
    value: str,
    source: str = "",
    confidence: str = "medium",
  ) -> None:
    if category not in self._data:
      self._data[category] = {}

    bucket = self._data[category]
    entry = {"value": value, "source": source, "confidence": confidence}

    if isinstance(bucket, dict) and key in bucket:
      old = bucket[key]
      if isinstance(old, dict) and old.get("value") != value:
        self._change_history.append({
          "category": category,
          "key": key,
          "old_value": old.get("value"),
          "new_value": value,
          "changed_at": datetime.now().isoformat(),
          "source": source,
        })
        logger.info("用户画像更新: %s.%s = %s → %s", category, key, old.get("value"), value)

    bucket[key] = entry
    self._maybe_persist()

  def set_simple(self, category: str, value: str, source: str = "") -> None:
    old = self._data.get(category)
    if isinstance(old, dict) and old.get("value") and old["value"] != value:
      self._change_history.append({
        "category": category,
        "old_value": old["value"],
        "new_value": value,
        "changed_at": datetime.now().isoformat(),
        "source": source,
      })

    self._data[category] = {"value": value, "source": source, "confidence": "high"}
    self._maybe_persist()

  def to_prompt(self) -> str:
    lines = ["【观众画像】"]
    empty = True

    for cat in _CATEGORIES:
      bucket = self._data.get(cat, {})
      if not bucket:
        continue

      if isinstance(bucket, dict) and "value" in bucket and isinstance(bucket["value"], str):
        lines.append(f"- {_category_label(cat)}：{bucket['value']}")
        empty = False
      elif isinstance(bucket, dict):
        for key, entry in bucket.items():
          if isinstance(entry, dict) and entry.get("value"):
            lines.append(f"- {_category_label(cat)}（{key}）：{entry['value']}")
            empty = False

    if empty:
      return ""
    return "\n".join(lines)

  def is_empty(self) -> bool:
    for bucket in self._data.values():
      if isinstance(bucket, dict):
        if "value" in bucket and isinstance(bucket["value"], str) and bucket["value"]:
          return False
        for v in bucket.values():
          if isinstance(v, dict) and v.get("value"):
            return False
    return True

  def clear(self) -> None:
    self._data = {cat: {} for cat in _CATEGORIES}
    self._change_history.clear()

  def debug_state(self) -> dict:
    return {
      "data": self._data,
      "change_history_count": len(self._change_history),
    }

  def _maybe_persist(self) -> None:
    if self._persist_path:
      try:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
          json.dumps(self._data, ensure_ascii=False, indent=2),
          encoding="utf-8",
        )
      except Exception as e:
        logger.error("持久化用户画像失败: %s", e)

  def _load_from_disk(self) -> None:
    try:
      self._data = json.loads(self._persist_path.read_text(encoding="utf-8"))
    except Exception as e:
      logger.error("加载用户画像失败: %s", e)


def _category_label(cat: str) -> str:
  labels = {
    "likes": "喜欢",
    "dislikes": "讨厌",
    "habits": "习惯",
    "occupation": "职业",
    "pets": "宠物",
    "catchphrases": "口头禅",
    "other": "其他",
  }
  return labels.get(cat, cat)
