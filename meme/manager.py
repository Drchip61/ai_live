"""
梗生命周期管理器
管理梗的识别、存储、演变和退休
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LifecycleStage(str, Enum):
  GROWING = "growing"
  MATURE = "mature"
  RETIRED = "retired"


_GROWING_DAYS = 14
_POSITIVE_THRESHOLD_FOR_MATURE = 3
_NEGATIVE_STREAK_FOR_RETIRE = 3


@dataclass
class Meme:
  id: str
  type: str
  content: str
  origin: str
  variants: list[str] = field(default_factory=list)
  created_at: str = ""
  lifecycle_stage: LifecycleStage = LifecycleStage.GROWING
  usage_count: int = 0
  last_used: Optional[str] = None
  user_reaction_positive: int = 0
  user_reaction_neutral: int = 0
  user_reaction_negative: int = 0

  def to_dict(self) -> dict:
    return {
      "id": self.id,
      "type": self.type,
      "content": self.content,
      "origin": self.origin,
      "variants": self.variants,
      "created_at": self.created_at,
      "lifecycle_stage": self.lifecycle_stage.value,
      "usage_count": self.usage_count,
      "last_used": self.last_used,
      "user_reaction_positive": self.user_reaction_positive,
      "user_reaction_neutral": self.user_reaction_neutral,
      "user_reaction_negative": self.user_reaction_negative,
    }

  @classmethod
  def from_dict(cls, d: dict) -> "Meme":
    return cls(
      id=d["id"],
      type=d.get("type", "callback"),
      content=d["content"],
      origin=d.get("origin", ""),
      variants=d.get("variants", []),
      created_at=d.get("created_at", ""),
      lifecycle_stage=LifecycleStage(d.get("lifecycle_stage", "growing")),
      usage_count=d.get("usage_count", 0),
      last_used=d.get("last_used"),
      user_reaction_positive=d.get("user_reaction_positive", 0),
      user_reaction_neutral=d.get("user_reaction_neutral", 0),
      user_reaction_negative=d.get("user_reaction_negative", 0),
    )


class MemeManager:
  """
  梗生命周期管理器

  负责梗的存储、生命周期流转（growing → mature → retired）、
  使用频率控制和用户反应跟踪。
  """

  def __init__(
    self,
    persist_path: Optional[Path] = None,
    seed_path: Optional[Path] = None,
  ) -> None:
    self._memes: dict[str, Meme] = {}
    self._next_id = 1
    self._persist_path = persist_path
    if persist_path and persist_path.exists():
      self._load_from_disk()

    if seed_path and seed_path.exists() and not self._memes:
      self._load_seeds(seed_path)

  def add(
    self,
    meme_type: str,
    content: str,
    origin: str,
    variants: Optional[list[str]] = None,
  ) -> Meme:
    """注册一个新梗"""
    for existing in self._memes.values():
      if existing.content == content:
        return existing

    meme_id = f"meme_{self._next_id:03d}"
    self._next_id += 1

    meme = Meme(
      id=meme_id,
      type=meme_type,
      content=content,
      origin=origin,
      variants=variants or [],
      created_at=datetime.now().strftime("%Y-%m-%d"),
      lifecycle_stage=LifecycleStage.GROWING,
    )
    self._memes[meme_id] = meme
    logger.info("新增梗: [%s] %s（来源：%s）", meme_type, content, origin)
    self._maybe_persist()
    return meme

  def record_usage(self, meme_id: str, user_reaction: str = "neutral") -> None:
    """
    记录梗被使用一次

    Args:
      meme_id: 梗 ID
      user_reaction: "positive" / "neutral" / "negative"
    """
    meme = self._memes.get(meme_id)
    if not meme:
      return

    meme.usage_count += 1
    meme.last_used = datetime.now().strftime("%Y-%m-%d")

    if user_reaction == "positive":
      meme.user_reaction_positive += 1
    elif user_reaction == "negative":
      meme.user_reaction_negative += 1
    else:
      meme.user_reaction_neutral += 1

    self._update_lifecycle(meme)
    self._maybe_persist()

  def get_active_memes(self) -> list[Meme]:
    """获取非退休状态的梗"""
    return [m for m in self._memes.values() if m.lifecycle_stage != LifecycleStage.RETIRED]

  def find_relevant(self, text: str) -> list[Meme]:
    """找到与文本相关的活跃梗"""
    relevant = []
    text_lower = text.lower()
    for meme in self.get_active_memes():
      if meme.content.lower() in text_lower:
        relevant.append(meme)
        continue
      for variant in meme.variants:
        if variant.lower() in text_lower:
          relevant.append(meme)
          break
    return relevant

  def should_use(self, meme: Meme) -> bool:
    """基于生命周期判断是否应该使用某个梗"""
    if meme.lifecycle_stage == LifecycleStage.RETIRED:
      return meme.usage_count % 10 == 0

    if meme.lifecycle_stage == LifecycleStage.GROWING:
      return True

    if meme.last_used:
      try:
        last = datetime.strptime(meme.last_used, "%Y-%m-%d")
        if (datetime.now() - last).days < 2:
          return False
      except ValueError:
        pass
    return True

  def to_prompt(self) -> str:
    """格式化活跃梗为 prompt 文本"""
    active = self.get_active_memes()
    if not active:
      return ""

    lines = ["【可用的梗和内部笑话】"]
    for meme in active:
      stage_label = {"growing": "成长期", "mature": "成熟期"}.get(
        meme.lifecycle_stage.value, ""
      )
      lines.append(f"- 「{meme.content}」（{meme.type}，{stage_label}）：{meme.origin}")
      if meme.variants:
        lines.append(f"  变体用法：{' / '.join(meme.variants[:3])}")
    return "\n".join(lines)

  def debug_state(self) -> dict:
    return {
      "total": len(self._memes),
      "active": len(self.get_active_memes()),
      "memes": [m.to_dict() for m in self._memes.values()],
    }

  def _update_lifecycle(self, meme: Meme) -> None:
    """根据使用数据更新梗的生命周期阶段"""
    if meme.lifecycle_stage == LifecycleStage.RETIRED:
      return

    recent_negative = meme.user_reaction_negative
    recent_total = meme.user_reaction_positive + meme.user_reaction_neutral + recent_negative
    if recent_total >= 3 and recent_negative >= _NEGATIVE_STREAK_FOR_RETIRE:
      meme.lifecycle_stage = LifecycleStage.RETIRED
      logger.info("梗退休: %s (连续负面反应)", meme.content)
      return

    if meme.lifecycle_stage == LifecycleStage.GROWING:
      if meme.created_at:
        try:
          created = datetime.strptime(meme.created_at, "%Y-%m-%d")
          age_days = (datetime.now() - created).days
          if age_days >= _GROWING_DAYS and meme.user_reaction_positive >= _POSITIVE_THRESHOLD_FOR_MATURE:
            meme.lifecycle_stage = LifecycleStage.MATURE
            logger.info("梗成熟: %s (使用 %d 天)", meme.content, age_days)
        except ValueError:
          pass

  def _maybe_persist(self) -> None:
    if self._persist_path:
      try:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self._memes.values()]
        self._persist_path.write_text(
          json.dumps(data, ensure_ascii=False, indent=2),
          encoding="utf-8",
        )
      except Exception as e:
        logger.error("持久化梗数据失败: %s", e)

  def _load_seeds(self, seed_path: Path) -> None:
    """从种子文件加载预置梗（仅当持久化数据为空时执行）"""
    try:
      data = json.loads(seed_path.read_text(encoding="utf-8"))
      for d in data:
        meme = Meme.from_dict(d)
        if not meme.created_at:
          meme.created_at = datetime.now().strftime("%Y-%m-%d")
        self._memes[meme.id] = meme
      if self._memes:
        max_num = max(
          int(mid.split("_")[1]) for mid in self._memes if mid.startswith("meme_")
        )
        self._next_id = max_num + 1
      logger.info("从种子文件加载了 %d 个预置梗", len(self._memes))
      self._maybe_persist()
    except Exception as e:
      logger.error("加载种子梗数据失败: %s", e)

  def _load_from_disk(self) -> None:
    try:
      data = json.loads(self._persist_path.read_text(encoding="utf-8"))
      for d in data:
        meme = Meme.from_dict(d)
        self._memes[meme.id] = meme
      if self._memes:
        max_num = max(
          int(mid.split("_")[1]) for mid in self._memes if mid.startswith("meme_")
        )
        self._next_id = max_num + 1
    except Exception as e:
      logger.error("加载梗数据失败: %s", e)
