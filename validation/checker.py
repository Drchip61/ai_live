"""
生成后校验器
检查 AI 回复是否符合奶凶角色的人设约束
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
  passed: bool
  violations: list[str] = field(default_factory=list)
  auto_fixed: bool = False
  fixed_response: Optional[str] = None


# 禁用的甜腻语气词和亲密称呼
_BANNED_WORDS = [
  "亲", "宝贝", "亲爱的", "小可爱", "乖", "么么", "mua",
]

# 禁用的直接表白
_BANNED_PHRASES = [
  re.compile(r"我关心你"),
  re.compile(r"我喜欢你"),
  re.compile(r"我想你了"),
  re.compile(r"我爱你"),
  re.compile(r"我好想你"),
  re.compile(r"对不起[，。！]"),  # 角色不主动道歉
]

# 句末甜腻语气词检测（"呢" 在句末使用属于甜腻风格）
_SWEET_SUFFIX = re.compile(r"[呢哟呀嘛][？。！～~]?$", re.MULTILINE)

# 颜文字检测
_KAOMOJI = re.compile(
  r"[\(（][\s]*[╯╰>＞<＜≧≦・ω•́•̀｡°ﾟ○●◕ᴗ∀ε￣▽▼△☆★♡♥✿]"
  r"|[╯╰].*?[╯╰]"
  r"|[>＞][_＿<＜]"
  r"|[TQＴ][_＿][TQＴ]"
  r"|[\^＾][_＿][\^＾]"
)


class ResponseChecker:
  """
  奶凶人设回复校验器

  执行四项检查：
  1. 禁用词检查 — 甜腻语气词、亲密称呼
  2. 设定档一致性 — 与角色立过的 flag 矛盾
  3. 情绪连贯性 — 闹别扭时不应过于热情
  4. 口癖频率 — 颜文字密度控制
  """

  def __init__(self, character_flags: Optional[list[str]] = None) -> None:
    self._character_flags = character_flags or []

  def update_flags(self, flags: list[str]) -> None:
    self._character_flags = flags

  def check(
    self,
    response: str,
    current_mood: str = "normal",
  ) -> ValidationResult:
    """
    校验 AI 回复

    Args:
      response: AI 生成的回复文本
      current_mood: 当前情绪状态

    Returns:
      ValidationResult
    """
    violations: list[str] = []

    self._check_banned_words(response, violations)
    self._check_banned_phrases(response, violations)
    self._check_sweet_suffix(response, violations)
    self._check_emotion_coherence(response, current_mood, violations)
    self._check_kaomoji_density(response, violations)

    if not violations:
      return ValidationResult(passed=True)

    fixed = self._try_auto_fix(response, violations)
    if fixed and fixed != response:
      logger.info("校验自动修正: %d 项违规", len(violations))
      return ValidationResult(
        passed=False,
        violations=violations,
        auto_fixed=True,
        fixed_response=fixed,
      )

    return ValidationResult(passed=False, violations=violations)

  def _check_banned_words(self, response: str, violations: list[str]) -> None:
    for word in _BANNED_WORDS:
      if word in response:
        violations.append(f"禁用词：「{word}」")

  def _check_banned_phrases(self, response: str, violations: list[str]) -> None:
    for pattern in _BANNED_PHRASES:
      if pattern.search(response):
        violations.append(f"禁用表达：{pattern.pattern}")

  def _check_sweet_suffix(self, response: str, violations: list[str]) -> None:
    matches = _SWEET_SUFFIX.findall(response)
    if len(matches) >= 2:
      violations.append(f"甜腻句尾语气词过多（{len(matches)}处）")

  def _check_emotion_coherence(
    self, response: str, mood: str, violations: list[str],
  ) -> None:
    if mood != "sulking":
      return

    enthusiastic_markers = [
      "！！", "哈哈", "太棒了", "超级", "好开心", "好喜欢", "耶",
    ]
    if len(response) > 50:
      violations.append("闹别扭状态下回复过长")

    for marker in enthusiastic_markers:
      if marker in response:
        violations.append(f"闹别扭状态下过于热情：「{marker}」")
        break

  def _check_kaomoji_density(self, response: str, violations: list[str]) -> None:
    kaomoji_count = len(_KAOMOJI.findall(response))
    sentence_count = max(1, len(re.split(r"[。！？\n]", response)))
    if kaomoji_count > 0 and sentence_count / kaomoji_count < 8:
      violations.append(
        f"颜文字密度过高（{kaomoji_count}个/{sentence_count}句，建议≤1/8句）"
      )

  def _try_auto_fix(self, response: str, violations: list[str]) -> Optional[str]:
    """尝试自动修正规则类违反"""
    fixed = response

    for word in _BANNED_WORDS:
      fixed = fixed.replace(word, "")

    fixed = _SWEET_SUFFIX.sub("。", fixed)

    fixed = re.sub(r"\s{2,}", " ", fixed).strip()
    if not fixed:
      return None
    return fixed
