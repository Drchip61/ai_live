"""
主播状态卡

维护日状态 / 场状态 / 瞬时反应三层动态状态，
注入 extra_context 供主 LLM 感知情绪惯性和行为连续性。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
  return datetime.now().isoformat()


@dataclass(frozen=True)
class StateCard:
  """主播状态卡（不可变，更新时返回新实例）"""

  # --- 日状态层 (开播初始化, 缓慢演变) ---
  daily_theme: str = ""
  energy: float = 0.7
  patience: float = 0.7
  current_obsession: str = ""

  # --- 场状态层 (本场直播, 中速变化) ---
  stream_phase: str = "开场"
  atmosphere: str = ""

  # --- 瞬时反应层 (1-3轮, 快速变化) ---
  undigested_emotion: str = ""
  near_term_goal: str = ""

  # --- 元数据 ---
  round_count: int = 0
  updated_at: str = field(default_factory=_now_iso)

  def to_prompt(self) -> str:
    """格式化为注入 extra_context 的文本"""
    lines = ["【当前状态】"]
    lines.append(
      "以下状态仅供感知自身倾向，弹幕永远优先，不要强行执行近期意图。"
    )
    if self.daily_theme:
      lines.append(f"今日主线：{self.daily_theme}")
    lines.append(f"精力：{self.energy:.2f} | 耐心：{self.patience:.2f}")
    if self.current_obsession:
      lines.append(f"执念：{self.current_obsession}")
    phase_atmo = self.stream_phase or "未知"
    if self.atmosphere:
      phase_atmo += f" | 氛围：{self.atmosphere}"
    lines.append(f"直播阶段：{phase_atmo}")
    if self.undigested_emotion:
      lines.append(f"未消化情绪：{self.undigested_emotion}")
    if self.near_term_goal:
      lines.append(f"近期意图：{self.near_term_goal}")
    return "\n".join(lines)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> StateCard:
    known_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    if "energy" in filtered:
      filtered["energy"] = _clamp01(filtered["energy"])
    if "patience" in filtered:
      filtered["patience"] = _clamp01(filtered["patience"])
    return cls(**filtered)

  @classmethod
  def from_json(cls, text: str) -> StateCard:
    return cls.from_dict(json.loads(text))

  def with_update(self, **kwargs: Any) -> StateCard:
    """返回更新指定字段后的新实例"""
    if "energy" in kwargs:
      kwargs["energy"] = _clamp01(kwargs["energy"])
    if "patience" in kwargs:
      kwargs["patience"] = _clamp01(kwargs["patience"])
    kwargs["updated_at"] = _now_iso()
    return replace(self, **kwargs)


def _clamp01(v: Any) -> float:
  try:
    return max(0.0, min(1.0, float(v)))
  except (TypeError, ValueError):
    return 0.5


# LLM 更新 prompt 中对各字段的说明（供 init / update 模板使用）
SCHEMA_DESCRIPTION = """\
{
  "daily_theme": "string — 今日主线情绪概括，如'有点困但聊起来就兴奋了'",
  "energy": "float 0.0-1.0 — 精力值，高=兴奋活跃，低=疲惫懒散",
  "patience": "float 0.0-1.0 — 耐心值，高=耐心倾听，低=容易不耐烦",
  "current_obsession": "string — 最近想聊/想做的事，可为空",
  "stream_phase": "string — 当前直播阶段：开场/热场/闲聊/高潮/收尾",
  "atmosphere": "string — 直播间氛围：轻松/热闹/被拱火/安慰观众/梗回收",
  "undigested_emotion": "string — 刚刚发生的、还没消化的情绪反应，可为空",
  "near_term_goal": "string — 接下来几分钟想做的事（倾向，不是指令），可为空"
}"""
