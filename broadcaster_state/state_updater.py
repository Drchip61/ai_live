"""
状态卡更新器

双轨更新：
- apply_event()  — 事件即时更新（规则引擎，同步）
- update_round() — 轮次异步更新（自然衰减 + 小 LLM 叙事 + 地板保护）
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from .state_card import StateCard, SCHEMA_DESCRIPTION, _clamp01

if TYPE_CHECKING:
  pass

logger = logging.getLogger(__name__)

_RULES_PATH = Path(__file__).parent / "update_rules.json"
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "state"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _load_rules(path: Path) -> dict[str, Any]:
  if not path.exists():
    logger.warning("规则文件不存在: %s，使用内置默认值", path)
    return {}
  return json.loads(path.read_text(encoding="utf-8"))


def _load_prompt_template(name: str) -> str:
  path = _PROMPTS_DIR / name
  if not path.exists():
    raise FileNotFoundError(f"状态 prompt 模板不存在: {path}")
  return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> Optional[dict]:
  """从 LLM 输出中提取 JSON（支持 ```json 代码块和裸 JSON）"""
  m = _JSON_BLOCK_RE.search(text)
  if m:
    try:
      return json.loads(m.group(1))
    except json.JSONDecodeError:
      pass
  first_brace = text.find("{")
  last_brace = text.rfind("}")
  if first_brace >= 0 and last_brace > first_brace:
    try:
      return json.loads(text[first_brace:last_brace + 1])
    except json.JSONDecodeError:
      pass
  return None


class StateUpdater:
  """
  状态卡更新器

  双轨机制：
  1. 规则引擎（同步）：事件增益、自然衰减、地板保护、delta clamp
  2. 小 LLM（异步）：叙事字段更新（daily_theme, atmosphere, emotion 等）
  """

  def __init__(
    self,
    model: BaseChatModel,
    rules_path: Optional[Path] = None,
  ) -> None:
    self._model = model
    rules = _load_rules(rules_path or _RULES_PATH)

    self._event_boosts: dict[str, dict[str, float]] = rules.get("event_boosts", {
      "gift": {"energy": 0.05, "patience": 0.03},
      "guard": {"energy": 0.15, "patience": 0.10},
      "super_chat": {"energy": 0.10, "patience": 0.08},
      "good_topic": {"energy": 0.08, "patience": 0.05},
    })
    floor = rules.get("floor", {})
    self._floor_energy: float = float(floor.get("energy", 0.25))
    self._floor_patience: float = float(floor.get("patience", 0.20))

    recovery = rules.get("recovery_rate", {})
    self._recovery_energy: float = float(recovery.get("energy", 0.05))
    self._recovery_patience: float = float(recovery.get("patience", 0.04))

    decay = rules.get("natural_decay", {})
    self._decay_energy: float = float(decay.get("energy_per_round", 0.02))
    self._decay_patience: float = float(decay.get("patience_per_round", 0.01))

    self._max_delta: float = float(rules.get("max_delta_per_round", 0.15))

  # ------------------------------------------------------------------
  # 路径 A：事件即时更新（同步，规则驱动）
  # ------------------------------------------------------------------

  def apply_event(self, card: StateCard, event_type: str) -> StateCard:
    """
    事件触发的即时数值更新。

    Args:
      card: 当前状态卡
      event_type: 事件类型 (gift / guard / super_chat / good_topic)

    Returns:
      更新后的新 StateCard
    """
    boost = self._event_boosts.get(event_type)
    if not boost:
      return card

    new_energy = _clamp01(card.energy + boost.get("energy", 0))
    new_patience = _clamp01(card.patience + boost.get("patience", 0))
    logger.info(
      "状态卡事件更新 [%s]: energy %.2f→%.2f, patience %.2f→%.2f",
      event_type, card.energy, new_energy, card.patience, new_patience,
    )
    return card.with_update(energy=new_energy, patience=new_patience)

  # ------------------------------------------------------------------
  # 路径 B：轮次异步更新（规则 + 小 LLM）
  # ------------------------------------------------------------------

  async def update_round(
    self,
    card: StateCard,
    comments_text: str,
    ai_response: str,
    stream_duration_minutes: float,
    topic_context: str = "",
  ) -> StateCard:
    """
    每轮结束后的异步更新。

    1. 规则层：自然衰减 + 地板保护
    2. LLM 层：叙事字段更新 + energy/patience 建议（经 clamp 后生效）

    Args:
      card: 当前状态卡
      comments_text: 本轮弹幕文本
      ai_response: 主播本轮回复
      stream_duration_minutes: 直播已进行分钟数
      topic_context: 当前话题上下文（可选）

    Returns:
      更新后的新 StateCard
    """
    decayed = self._apply_decay(card)
    floored = self._enforce_floor(decayed)

    try:
      llm_card = await self._llm_update(
        floored, comments_text, ai_response,
        stream_duration_minutes, topic_context,
      )
      merged = self._merge_llm_result(floored, llm_card)
    except Exception as e:
      logger.warning("状态卡 LLM 更新失败，保持规则更新结果: %s", e)
      merged = floored

    final = self._enforce_floor(merged)
    return final.with_update(round_count=card.round_count + 1)

  # ------------------------------------------------------------------
  # 开播初始化
  # ------------------------------------------------------------------

  async def init_daily_state(
    self,
    persona_name: str,
    persona_summary: str,
    recent_memories: str = "",
    time_of_day: str = "",
  ) -> StateCard:
    """
    开播时生成今日初始状态卡。

    Args:
      persona_name: 角色名
      persona_summary: 角色简介
      recent_memories: 近期记忆摘要
      time_of_day: 当前时间描述

    Returns:
      初始 StateCard
    """
    try:
      template = _load_prompt_template("init_daily.txt")
      prompt = template.format(
        persona_name=persona_name,
        persona_summary=persona_summary,
        recent_memories=recent_memories or "（无近期记忆）",
        time_of_day=time_of_day,
        schema_description=SCHEMA_DESCRIPTION,
      )
      result = await self._model.ainvoke(prompt)
      text = result.content if hasattr(result, "content") else str(result)
      data = _extract_json(text)
      if data:
        card = StateCard.from_dict(data)
        logger.info("状态卡初始化成功: %s", card.to_dict())
        return card
      logger.warning("状态卡初始化 JSON 解析失败，使用默认值")
    except Exception as e:
      logger.error("状态卡初始化失败: %s", e)

    return StateCard(daily_theme="精神不错，准备开播")

  # ------------------------------------------------------------------
  # 内部方法
  # ------------------------------------------------------------------

  def _apply_decay(self, card: StateCard) -> StateCard:
    """自然衰减"""
    return card.with_update(
      energy=card.energy - self._decay_energy,
      patience=card.patience - self._decay_patience,
    )

  def _enforce_floor(self, card: StateCard) -> StateCard:
    """地板保护：低于阈值时自动回升"""
    new_energy = card.energy
    new_patience = card.patience

    if new_energy < self._floor_energy:
      new_energy = min(
        self._floor_energy,
        new_energy + self._recovery_energy,
      )
    if new_patience < self._floor_patience:
      new_patience = min(
        self._floor_patience,
        new_patience + self._recovery_patience,
      )

    if new_energy != card.energy or new_patience != card.patience:
      return card.with_update(energy=new_energy, patience=new_patience)
    return card

  async def _llm_update(
    self,
    card: StateCard,
    comments_text: str,
    ai_response: str,
    stream_duration_minutes: float,
    topic_context: str,
  ) -> Optional[StateCard]:
    """调用小 LLM 更新叙事字段"""
    template = _load_prompt_template("update_state.txt")

    state_json = json.dumps({
      "daily_theme": card.daily_theme,
      "energy": round(card.energy, 2),
      "patience": round(card.patience, 2),
      "current_obsession": card.current_obsession,
      "stream_phase": card.stream_phase,
      "atmosphere": card.atmosphere,
      "undigested_emotion": card.undigested_emotion,
      "near_term_goal": card.near_term_goal,
    }, ensure_ascii=False, indent=2)

    prompt = template.format(
      current_state_json=state_json,
      stream_duration_minutes=f"{stream_duration_minutes:.0f}",
      comments_text=comments_text or "（本轮无弹幕）",
      ai_response=ai_response or "（未回复）",
      topic_context=topic_context or "（无话题上下文）",
    )

    result = await self._model.ainvoke(prompt)
    text = result.content if hasattr(result, "content") else str(result)
    data = _extract_json(text)
    if data is None:
      logger.warning("状态卡更新 JSON 解析失败: %s", text[:200])
      return None
    return StateCard.from_dict(data)

  def _merge_llm_result(
    self,
    base: StateCard,
    llm_card: Optional[StateCard],
  ) -> StateCard:
    """合并 LLM 结果，对数值字段做 delta clamp"""
    if llm_card is None:
      return base

    new_energy = self._clamp_delta(base.energy, llm_card.energy)
    new_patience = self._clamp_delta(base.patience, llm_card.patience)

    return base.with_update(
      daily_theme=llm_card.daily_theme or base.daily_theme,
      energy=new_energy,
      patience=new_patience,
      current_obsession=llm_card.current_obsession,
      stream_phase=llm_card.stream_phase or base.stream_phase,
      atmosphere=llm_card.atmosphere or base.atmosphere,
      undigested_emotion=llm_card.undigested_emotion,
      near_term_goal=llm_card.near_term_goal,
    )

  def _clamp_delta(self, old: float, new: float) -> float:
    """限制单轮变化幅度"""
    delta = new - old
    clamped = max(-self._max_delta, min(self._max_delta, delta))
    return _clamp01(old + clamped)

  def debug_state(self) -> dict:
    return {
      "event_boosts": self._event_boosts,
      "floor": {"energy": self._floor_energy, "patience": self._floor_patience},
      "recovery_rate": {"energy": self._recovery_energy, "patience": self._recovery_patience},
      "natural_decay": {"energy": self._decay_energy, "patience": self._decay_patience},
      "max_delta_per_round": self._max_delta,
    }
