"""
情感银行（好感度系统）
维护隐藏的好感度数值，影响角色行为温度
"""

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AffectionTier(str, Enum):
  HIGH = "high"
  MEDIUM = "medium"
  LOW = "low"


_TIER_THRESHOLDS = {
  AffectionTier.HIGH: 70,
  AffectionTier.MEDIUM: 30,
}

_TIER_PROMPT_HINTS = {
  AffectionTier.HIGH: (
    "【好感度：高】你对观众的态度偏暖：毒舌里夹带更多真心话，"
    "偶尔主动分享自己的心事，吐槽时会带着宠溺感。"
  ),
  AffectionTier.MEDIUM: (
    "【好感度：中】标准傲娇模式：正常比例的嘴硬和真心。"
  ),
  AffectionTier.LOW: (
    "【好感度：低】你现在话更少，更冷淡，但仍不离开。"
    "减少主动展开话题的意愿，回复偏简短。"
  ),
}

_CARE_PATTERNS = re.compile(
  r"(你还好吗|你[吃喝]了吗|注意[身休]|早点[睡休]|别太累|照顾好自己|"
  r"你今天怎么样|关心你|担心你|心疼你)"
)
_REMEMBER_PATTERNS = re.compile(
  r"(你[之上]次说|你说过|记得你说|你不是说|你提到过)"
)
_NEGATIVE_PATTERNS = re.compile(
  r"(你好烦|讨厌你|闭嘴|滚|无聊死了|不想理你|你真没用)"
)
_STABLE_INTERACTION_BONUS_ROUNDS = 20


class AffectionBank:
  """
  好感度银行

  维护 0-100 的好感度数值（用户不可见），根据互动事件动态调整。
  对外提供 tier（高/中/低）和对应的 prompt 行为指引。
  """

  def __init__(self, initial: float = 50.0) -> None:
    self._value = max(0.0, min(100.0, initial))
    self._interaction_count = 0

  @property
  def value(self) -> float:
    return self._value

  @property
  def tier(self) -> AffectionTier:
    if self._value >= _TIER_THRESHOLDS[AffectionTier.HIGH]:
      return AffectionTier.HIGH
    if self._value >= _TIER_THRESHOLDS[AffectionTier.MEDIUM]:
      return AffectionTier.MEDIUM
    return AffectionTier.LOW

  def to_prompt(self) -> str:
    return _TIER_PROMPT_HINTS.get(self.tier, "")

  def process_interaction(
    self,
    user_text: str,
    ai_response: str,
    meme_caught: bool = False,
  ) -> float:
    """
    处理一次互动，更新好感度

    Args:
      user_text: 用户输入
      ai_response: AI 回复
      meme_caught: 用户是否接住了梗

    Returns:
      好感度变化值（正为加分，负为减分）
    """
    delta = 0.0

    if _CARE_PATTERNS.search(user_text):
      delta += 3.0
      logger.debug("好感度 +3: 用户主动关心")

    if _REMEMBER_PATTERNS.search(user_text):
      delta += 4.0
      logger.debug("好感度 +4: 用户记得角色说过的话")

    if meme_caught:
      delta += 2.0
      logger.debug("好感度 +2: 用户接住了梗")

    if _NEGATIVE_PATTERNS.search(user_text):
      delta -= 5.0
      logger.debug("好感度 -5: 用户否定性评价")

    self._interaction_count += 1
    if self._interaction_count % _STABLE_INTERACTION_BONUS_ROUNDS == 0:
      delta += 2.0
      logger.debug("好感度 +2: 稳定互动奖励（%d轮）", self._interaction_count)

    if delta != 0:
      self._apply_delta(delta)

    return delta

  def apply_absence_penalty(self, silence_hours: float) -> float:
    """长时间未互动的衰减"""
    if silence_hours < 24:
      return 0.0
    penalty = min(10.0, silence_hours / 24 * 2.0)
    self._apply_delta(-penalty)
    logger.debug("好感度 -%.1f: 长时间未互动（%.0f小时）", penalty, silence_hours)
    return -penalty

  def _apply_delta(self, delta: float) -> None:
    old = self._value
    self._value = max(0.0, min(100.0, self._value + delta))
    if abs(self._value - old) > 0.01:
      logger.info("好感度: %.1f → %.1f (%+.1f)", old, self._value, delta)

  def debug_state(self) -> dict:
    return {
      "value": round(self._value, 1),
      "tier": self.tier.value,
      "interaction_count": self._interaction_count,
    }
