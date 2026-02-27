"""
角色设定档层
存储角色在对话中"立过的 flag"（偏好、观点、承诺、起过的外号）
这是人设一致性的底线，极少更新
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CharacterProfileLayer:
  """
  角色设定档存储层

  存储角色自身在对话中表达过的偏好、观点、承诺和给用户起的外号。
  写入后长期保持，极少更新。
  """

  def __init__(self, persist_path: Optional[Path] = None) -> None:
    self._data: dict = {
      "preferences": [],
      "claims": [],
      "nicknames_given": [],
    }
    self._persist_path = persist_path
    if persist_path and persist_path.exists():
      self._load_from_disk()

  @property
  def preferences(self) -> list[dict]:
    return self._data.get("preferences", [])

  @property
  def claims(self) -> list[dict]:
    return self._data.get("claims", [])

  @property
  def nicknames(self) -> list[dict]:
    return self._data.get("nicknames_given", [])

  def add_preference(self, content: str, source: str = "", weight: float = 1.0) -> None:
    if self._has_similar("preferences", content):
      return
    self._data["preferences"].append({
      "content": content,
      "source": source,
      "weight": weight,
      "created_at": datetime.now().isoformat(),
    })
    logger.info("角色设定档 +偏好: %s", content)
    self._maybe_persist()

  def add_claim(self, content: str, source: str = "", weight: float = 0.9) -> None:
    if self._has_similar("claims", content):
      return
    self._data["claims"].append({
      "content": content,
      "source": source,
      "weight": weight,
      "created_at": datetime.now().isoformat(),
    })
    logger.info("角色设定档 +声明: %s", content)
    self._maybe_persist()

  def add_nickname(
    self, target: str, nickname: str, origin: str = "", source: str = "",
  ) -> None:
    for existing in self._data["nicknames_given"]:
      if existing.get("nickname") == nickname:
        return
    self._data["nicknames_given"].append({
      "target": target,
      "nickname": nickname,
      "origin": origin,
      "source": source,
      "created_at": datetime.now().isoformat(),
    })
    logger.info("角色设定档 +外号: %s → %s", target, nickname)
    self._maybe_persist()

  def get_all_flags(self) -> list[str]:
    flags = []
    for p in self.preferences:
      flags.append(p.get("content", ""))
    for c in self.claims:
      flags.append(c.get("content", ""))
    return [f for f in flags if f]

  def find_relevant_flags(self, text: str) -> list[str]:
    relevant = []
    text_lower = text.lower()
    for flag in self.get_all_flags():
      flag_lower = flag.lower()
      if any(kw in text_lower for kw in _extract_keywords(flag_lower)):
        relevant.append(flag)
    return relevant

  def to_prompt(self) -> str:
    parts = ["【角色设定档 — 我立过的 flag】"]
    empty = True

    if self.preferences:
      parts.append("偏好：")
      for p in self.preferences:
        parts.append(f"  - {p.get('content', '')}")
      empty = False

    if self.claims:
      parts.append("声明/承诺：")
      for c in self.claims:
        parts.append(f"  - {c.get('content', '')}")
      empty = False

    if self.nicknames:
      parts.append("给观众起的外号：")
      for n in self.nicknames:
        parts.append(f"  - {n.get('target', '?')} → 「{n.get('nickname', '')}」（因为{n.get('origin', '?')}）")
      empty = False

    if empty:
      return ""
    return "\n".join(parts)

  def is_empty(self) -> bool:
    return (
      not self.preferences
      and not self.claims
      and not self.nicknames
    )

  def clear(self) -> None:
    self._data = {"preferences": [], "claims": [], "nicknames_given": []}

  def debug_state(self) -> dict:
    return {
      "preferences_count": len(self.preferences),
      "claims_count": len(self.claims),
      "nicknames_count": len(self.nicknames),
      "data": self._data,
    }

  def _has_similar(self, category: str, content: str) -> bool:
    for item in self._data.get(category, []):
      if item.get("content", "").strip() == content.strip():
        return True
    return False

  def _maybe_persist(self) -> None:
    if self._persist_path:
      try:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(
          json.dumps(self._data, ensure_ascii=False, indent=2),
          encoding="utf-8",
        )
      except Exception as e:
        logger.error("持久化角色设定档失败: %s", e)

  def _load_from_disk(self) -> None:
    try:
      self._data = json.loads(self._persist_path.read_text(encoding="utf-8"))
    except Exception as e:
      logger.error("加载角色设定档失败: %s", e)


def _extract_keywords(text: str) -> list[str]:
  words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text)
  return words
