"""
情绪状态机
管理角色的情绪状态流转，影响回复风格
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Mood(str, Enum):
  NORMAL = "normal"
  COMPETITIVE = "competitive"
  SULKING = "sulking"
  PROUD = "proud"
  SOFT = "soft"


# 每种情绪在无外部刺激时自动回归 normal 的最大轮次
_AUTO_DECAY_ROUNDS = {
  Mood.COMPETITIVE: 6,
  Mood.SULKING: 8,
  Mood.PROUD: 4,
  Mood.SOFT: 2,
}

_MOOD_PROMPT_HINTS: dict[Mood, str] = {
  Mood.NORMAL: (
    "你现在处于日常傲娇状态：轻微嘴硬+偶尔吐槽，正常对话节奏，"
    "愿意聊天但嘴上不承认。"
  ),
  Mood.COMPETITIVE: (
    "你现在好胜心爆发了：认真论证自己是对的，话变多，语气更强硬但不要攻击性，"
    "一定要找到证据说服对方。"
  ),
  Mood.SULKING: (
    "你现在在闹别扭：回复变短，语气变冷，不主动展开话题。"
    "只用「哦」「嗯」「随便」「……」这类简短回复。"
    "不要表现出热情。如果对方道歉或说好话，可以找台阶下来恢复。"
  ),
  Mood.PROUD: (
    "你现在很得意：因为之前的判断被证明是对的。"
    "语气得意但不过分，可以用「看吧，我说什么来着」这类表达。"
  ),
  Mood.SOFT: (
    "你现在偶尔真情流露：话变柔软，但仍然不完全坦白，"
    "保留一点点傲娇尾巴。这种状态极其短暂，说完一两句就要收回来。"
  ),
}


@dataclass
class EmotionState:
  """当前情绪快照"""
  mood: Mood = Mood.NORMAL
  intensity: float = 0.5
  trigger: Optional[str] = None
  duration_rounds: int = 0
  recovery_condition: Optional[str] = None

  def to_prompt(self) -> str:
    hint = _MOOD_PROMPT_HINTS.get(self.mood, "")
    parts = [f"【当前情绪状态：{self.mood.value}（强度 {self.intensity:.1f}）】"]
    if hint:
      parts.append(hint)
    if self.trigger:
      parts.append(f"触发原因：{self.trigger}")
    return "\n".join(parts)

  def to_dict(self) -> dict:
    return {
      "current_mood": self.mood.value,
      "mood_intensity": round(self.intensity, 2),
      "trigger": self.trigger,
      "duration_rounds": self.duration_rounds,
      "recovery_condition": self.recovery_condition,
    }


class EmotionMachine:
  """
  情绪状态机

  管理 normal / competitive / sulking / proud / soft 五种状态的流转，
  提供 tick()（每轮自动衰减）和 transition()（外部触发转换）两个驱动接口。
  """

  def __init__(self) -> None:
    self._state = EmotionState()

  @property
  def state(self) -> EmotionState:
    return self._state

  @property
  def mood(self) -> Mood:
    return self._state.mood

  def tick(self) -> None:
    """每轮对话调用一次，处理自动衰减"""
    if self._state.mood == Mood.NORMAL:
      return

    self._state.duration_rounds += 1
    max_rounds = _AUTO_DECAY_ROUNDS.get(self._state.mood, 5)

    if self._state.duration_rounds >= max_rounds:
      logger.info(
        "情绪自动衰减: %s → normal (持续 %d 轮)",
        self._state.mood.value, self._state.duration_rounds,
      )
      self._reset_to_normal()
      return

    decay = 0.1
    self._state.intensity = max(0.1, self._state.intensity - decay)

  def transition(self, target: Mood, trigger: str, intensity: float = 0.7) -> None:
    """
    外部触发状态转换

    Args:
      target: 目标情绪
      trigger: 触发原因描述
      intensity: 初始强度 (0-1)
    """
    if target == self._state.mood:
      self._state.intensity = min(1.0, self._state.intensity + 0.2)
      self._state.trigger = trigger
      return

    old = self._state.mood
    self._state = EmotionState(
      mood=target,
      intensity=min(1.0, max(0.1, intensity)),
      trigger=trigger,
      duration_rounds=0,
      recovery_condition=_recovery_for(target),
    )
    logger.info("情绪转换: %s → %s (触发: %s)", old.value, target.value, trigger)

  def force_normal(self) -> None:
    """强制重置为 normal"""
    self._reset_to_normal()

  def _reset_to_normal(self) -> None:
    self._state = EmotionState(
      mood=Mood.NORMAL,
      intensity=0.5,
    )

  def debug_state(self) -> dict:
    return self._state.to_dict()


def _recovery_for(mood: Mood) -> Optional[str]:
  mapping = {
    Mood.SULKING: "观众道歉或说好话",
    Mood.COMPETITIVE: "争论结束或话题转移",
    Mood.PROUD: "自然衰减",
    Mood.SOFT: "自动回归（极短暂）",
  }
  return mapping.get(mood)
