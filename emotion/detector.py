"""
情绪触发检测器
分析用户输入，判断是否应触发情绪状态转换
"""

import re
from typing import Optional

from .state import Mood


_SULKING_PATTERNS = [
  re.compile(r"(你好烦|好无聊|烦死了|不想.*聊|闭嘴|滚|安静|别说了)"),
  re.compile(r"(讨厌你|不喜欢你|你真烦|无聊)"),
]

_COMPETITIVE_PATTERNS = [
  re.compile(r"(你[说讲]错了|不对吧|你确定|我不同意|你搞错了|胡说|瞎说)"),
  re.compile(r"(才不是|明明不是|我觉得你错了|反驳|不服)"),
]

_PROUD_PATTERNS = [
  re.compile(r"(你[说讲]得?对|你是对的|确实|服了|你赢了|好吧你说的对|你厉害)"),
]

_SOFT_PATTERNS = [
  re.compile(r"(好久没[来见]|想你了|终于来了|一直[在等]|谢谢你|你真好)"),
  re.compile(r"(对不起.*我.*[不没]好|遇到.*困难|心情.*[不很]好|难过|伤心)"),
]

_RECOVERY_PATTERNS = [
  re.compile(r"(对不起|我错了|别生气|sorry|抱歉|开玩笑的|逗你的|你不[烦无])"),
]

_POSITIVE_REACTION = re.compile(
  r"(哈哈|笑死|太好笑|233|lol|hhh|草|xswl|乐|绝了|好家伙|6{2,})"
)


class EmotionTriggerDetector:
  """
  基于规则的情绪触发检测器

  分析用户文本，返回建议的目标情绪和触发原因。
  若无触发则返回 None，交由 EmotionMachine 的 tick() 自然衰减。
  """

  def detect(
    self,
    user_text: str,
    current_mood: Mood,
  ) -> Optional[tuple[Mood, str]]:
    """
    检测用户输入是否触发情绪转换

    Args:
      user_text: 用户输入文本
      current_mood: 当前情绪状态

    Returns:
      (target_mood, trigger_reason) 或 None
    """
    if not user_text.strip():
      return None

    text = user_text.strip()

    if current_mood == Mood.SULKING:
      if self._match_any(text, _RECOVERY_PATTERNS):
        return Mood.NORMAL, "观众道歉/说好话，闹别扭恢复"
      return None

    if self._match_any(text, _SULKING_PATTERNS):
      return Mood.SULKING, f"观众说了伤人的话：{text[:30]}"

    if self._match_any(text, _COMPETITIVE_PATTERNS):
      return Mood.COMPETITIVE, f"观众质疑/反驳：{text[:30]}"

    if current_mood == Mood.COMPETITIVE:
      if self._match_any(text, _PROUD_PATTERNS):
        return Mood.PROUD, "观众认输/认同，好胜心得到满足"

    if self._match_any(text, _SOFT_PATTERNS):
      return Mood.SOFT, f"观众触动了真情：{text[:30]}"

    return None

  def has_positive_reaction(self, text: str) -> bool:
    """检测文本是否包含正面情绪反应（笑、夸等）"""
    return bool(_POSITIVE_REACTION.search(text))

  @staticmethod
  def _match_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)
