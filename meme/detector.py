"""
梗识别检测器
分析对话，识别潜在的梗信号
"""

import re
from dataclasses import dataclass
from typing import Optional


_POSITIVE_REACTION = re.compile(
  r"(哈哈|笑死|太好笑|233|lol|hhh|草|xswl|乐|好家伙|6{2,}|笑哭|绝了)"
)

_NICKNAME_PATTERN = re.compile(
  r"(叫[你他她它]|给[你他她它].*外号|以后.*叫|称呼.*为|封[你他她它]为|"
  r"[你他她它]就是.*[王帝神])"
)

_CATCHPHRASE_REPEAT_THRESHOLD = 3


@dataclass
class MemeSignal:
  """梗信号"""
  meme_type: str
  content: str
  origin: str
  confidence: float


class MemeDetector:
  """
  梗检测器

  分析 AI 回复和用户反应，识别有潜力成为梗的内容。
  信号类型：
  - 用户正面情绪反应（笑、夸）
  - 用户主动复述角色的表达
  - 角色给用户起外号
  - 角色翻车/被打脸的瞬间
  """

  def __init__(self) -> None:
    self._user_phrase_counts: dict[str, int] = {}

  def detect_from_exchange(
    self,
    user_text: str,
    ai_response: str,
    user_reaction: Optional[str] = None,
  ) -> list[MemeSignal]:
    """
    从一次对话交换中检测梗信号

    Args:
      user_text: 用户输入
      ai_response: AI 回复
      user_reaction: 用户对 AI 回复的后续反应（下一轮的用户输入）

    Returns:
      检测到的梗信号列表
    """
    signals: list[MemeSignal] = []

    nickname_signal = self._detect_nickname(ai_response)
    if nickname_signal:
      signals.append(nickname_signal)

    if user_reaction and _POSITIVE_REACTION.search(user_reaction):
      key_phrase = self._extract_key_phrase(ai_response)
      if key_phrase:
        signals.append(MemeSignal(
          meme_type="catchphrase",
          content=key_phrase,
          origin=f"观众对「{key_phrase}」表现出正面反应",
          confidence=0.6,
        ))

    if user_reaction:
      echo = self._detect_echo(ai_response, user_reaction)
      if echo:
        signals.append(MemeSignal(
          meme_type="callback",
          content=echo,
          origin=f"观众主动复述了「{echo}」",
          confidence=0.8,
        ))

    catchphrase = self._detect_user_catchphrase(user_text)
    if catchphrase:
      signals.append(catchphrase)

    return signals

  def _detect_nickname(self, ai_response: str) -> Optional[MemeSignal]:
    if _NICKNAME_PATTERN.search(ai_response):
      match = re.search(r"[「「](.+?)[」」]|叫[你他她](.{1,6})[，。！]", ai_response)
      if match:
        nickname = match.group(1) or match.group(2)
        return MemeSignal(
          meme_type="nickname",
          content=nickname.strip(),
          origin=f"角色起的外号：{nickname}",
          confidence=0.7,
        )
    return None

  def _detect_echo(self, ai_response: str, user_reaction: str) -> Optional[str]:
    """检测用户是否复述了 AI 的某个短语"""
    ai_phrases = re.findall(r"[\u4e00-\u9fff]{3,8}", ai_response)
    for phrase in ai_phrases:
      if phrase in user_reaction and len(phrase) >= 3:
        return phrase
    return None

  def _detect_user_catchphrase(self, user_text: str) -> Optional[MemeSignal]:
    """检测用户是否重复使用某个短语（口头禅）"""
    phrases = re.findall(r"[\u4e00-\u9fff]{2,6}", user_text)
    for phrase in phrases:
      self._user_phrase_counts[phrase] = self._user_phrase_counts.get(phrase, 0) + 1
      if self._user_phrase_counts[phrase] == _CATCHPHRASE_REPEAT_THRESHOLD:
        return MemeSignal(
          meme_type="catchphrase",
          content=phrase,
          origin=f"观众的口头禅：总是说「{phrase}」",
          confidence=0.5,
        )
    return None

  @staticmethod
  def _extract_key_phrase(text: str) -> Optional[str]:
    """提取文本中最有趣的短语"""
    sentences = re.split(r"[。！？\n]", text)
    for s in sentences:
      s = s.strip()
      if 4 <= len(s) <= 20 and not s.startswith("#["):
        return s
    return None
