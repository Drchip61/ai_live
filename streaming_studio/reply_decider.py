"""
回复决策器
两阶段判断主播是否应该回复：规则快筛（免费） + LLM 精判（Haiku）
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from .config import ReplyDeciderConfig
from .models import Comment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplyDecision:
  """回复决策结果"""
  should_reply: bool
  urgency: float
  reason: str
  phase: str  # "rule" or "llm"


class ReplyDecider:
  """
  两阶段回复决策器

  Phase 1（规则快筛，免费）：
    处理明确场景——必须回复（提问、高活跃）或建议跳过（纯反应词、刷屏）

  Phase 2（LLM 精判，Haiku）：
    当规则无法决定时，用轻量 LLM 综合弹幕内容、场景描述和沉默时长做判断
  """

  def __init__(
    self,
    config: Optional[ReplyDeciderConfig] = None,
    llm_model: Optional[BaseChatModel] = None,
    judge_prompt: str = "",
  ):
    self.config = config or ReplyDeciderConfig()
    self._llm = llm_model
    self._judge_prompt = judge_prompt
    self._skip_set = set(p.lower() for p in self.config.skip_patterns)

  def rule_check(
    self,
    old_comments: list[Comment],
    new_comments: list[Comment],
  ) -> Optional[ReplyDecision]:
    """
    Phase 1: 规则快筛

    Returns:
      明确决策时返回 ReplyDecision，不确定时返回 None
    """
    all_comments = old_comments + new_comments

    if not all_comments:
      return ReplyDecision(False, 0, "无弹幕", "rule")

    # 优先弹幕（手动输入）→ 必须回复
    if any(c.priority for c in all_comments):
      return ReplyDecision(True, 9, "有优先弹幕", "rule")

    # 新弹幕数量超过阈值 → 直接回复（聊天很活跃）
    if len(new_comments) >= self.config.must_reply_comment_count:
      return ReplyDecision(True, 8, f"新弹幕数量多({len(new_comments)}条)", "rule")

    # 包含提问（问号） → 必须回复
    for c in new_comments:
      if "?" in c.content or "？" in c.content:
        stripped = c.content.replace("?", "").replace("？", "").strip()
        if len(stripped) >= 2:
          return ReplyDecision(True, 8, f"观众提问: {c.content[:20]}", "rule")

    # 检查是否全是低质量内容
    low_quality_count = 0
    for c in new_comments:
      content = c.content.strip().lower()
      is_low = (
        len(content) <= self.config.min_quality_length
        or content in self._skip_set
        or self._is_repetitive(content)
      )
      if is_low:
        low_quality_count += 1

    if new_comments and low_quality_count == len(new_comments):
      return ReplyDecision(False, 1, "全部为低质量弹幕", "rule")

    # 不确定 → 交给 Phase 2
    return None

  async def llm_judge(
    self,
    old_comments: list[Comment],
    new_comments: list[Comment],
    scene_description: Optional[str] = None,
    silence_seconds: float = 0,
  ) -> ReplyDecision:
    """
    Phase 2: LLM 精判

    用轻量模型判断当前弹幕是否值得回复
    """
    if self._llm is None:
      return ReplyDecision(True, 5, "无LLM可用，默认回复", "llm")

    parts = []
    if scene_description:
      parts.append(f"[当前画面] {scene_description}")
    if silence_seconds > 0:
      parts.append(f"[沉默时长] 距上次回复已过 {int(silence_seconds)} 秒")

    if old_comments:
      lines = [f"  {c.nickname}: {c.content}" for c in old_comments[-5:]]
      parts.append("[旧弹幕]\n" + "\n".join(lines))

    if new_comments:
      lines = [f"  {c.nickname}: {c.content}" for c in new_comments]
      parts.append("[新弹幕]\n" + "\n".join(lines))

    user_text = "\n\n".join(parts)

    messages = [
      SystemMessage(content=self._judge_prompt),
      HumanMessage(content=user_text),
    ]

    try:
      result = await self._llm.ainvoke(messages)
      text = result.content if hasattr(result, "content") else str(result)
      return self._parse_judge_response(text)
    except Exception as e:
      logger.error("LLM 精判调用失败: %s", e)
      return ReplyDecision(True, 5, f"精判异常({e})，默认回复", "llm")

  async def should_reply(
    self,
    old_comments: list[Comment],
    new_comments: list[Comment],
    scene_description: Optional[str] = None,
    last_reply_time: Optional[datetime] = None,
  ) -> ReplyDecision:
    """
    完整两阶段决策

    Args:
      old_comments: 上次回复前的弹幕
      new_comments: 新弹幕
      scene_description: 当前场景描述（VLM 模式）
      last_reply_time: 上次回复时间（用于计算沉默时长）
    """
    # Phase 1
    decision = self.rule_check(old_comments, new_comments)
    if decision is not None:
      logger.info("回复决策[规则]: %s (urgency=%.0f, %s)",
                  "回复" if decision.should_reply else "跳过",
                  decision.urgency, decision.reason)
      return decision

    # Phase 2
    silence = 0.0
    if last_reply_time:
      silence = (datetime.now() - last_reply_time).total_seconds()

    decision = await self.llm_judge(
      old_comments, new_comments, scene_description, silence,
    )
    logger.info("回复决策[LLM]: %s (urgency=%.0f, %s)",
                "回复" if decision.should_reply else "跳过",
                decision.urgency, decision.reason)
    return decision

  async def should_proactive_speak(
    self,
    prev_scene: Optional[str],
    current_scene: Optional[str],
    silence_seconds: float,
  ) -> ReplyDecision:
    """
    判断是否应该主动发言（无弹幕时，基于画面变化）

    Args:
      prev_scene: 上一次场景描述
      current_scene: 当前场景描述
      silence_seconds: 沉默时长（秒）
    """
    if silence_seconds < self.config.proactive_silence_threshold:
      return ReplyDecision(False, 0, "沉默时间不足", "rule")

    if not current_scene or not prev_scene:
      return ReplyDecision(False, 0, "无场景信息", "rule")

    if current_scene == prev_scene:
      return ReplyDecision(False, 0, "场景无变化", "rule")

    # 用 LLM 判断场景变化是否有意义
    if self._llm is None:
      return ReplyDecision(True, 5, "场景变化，默认发言", "rule")

    prompt = (
      f"[上一次画面] {prev_scene}\n"
      f"[当前画面] {current_scene}\n"
      f"[沉默时长] {int(silence_seconds)} 秒\n\n"
      f"画面是否发生了值得主播评论的重要变化？"
    )
    messages = [
      SystemMessage(content=self._judge_prompt),
      HumanMessage(content=prompt),
    ]
    try:
      result = await self._llm.ainvoke(messages)
      text = result.content if hasattr(result, "content") else str(result)
      decision = self._parse_judge_response(text)
      logger.info("主动发言决策: %s (urgency=%.0f, %s)",
                  "发言" if decision.should_reply else "沉默",
                  decision.urgency, decision.reason)
      return decision
    except Exception as e:
      logger.error("主动发言判断失败: %s", e)
      return ReplyDecision(False, 0, f"判断异常({e})", "llm")

  def _parse_judge_response(self, text: str) -> ReplyDecision:
    """解析 LLM 精判的 JSON 响应"""
    text = text.strip()

    # 尝试提取 JSON（可能被包在 markdown code block 中）
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
      try:
        data = json.loads(json_match.group())
        reply = bool(data.get("reply", True))
        urgency = float(data.get("urgency", 5))

        if urgency < self.config.llm_judge_urgency_threshold:
          reply = False

        return ReplyDecision(
          should_reply=reply,
          urgency=urgency,
          reason=data.get("reason", "LLM判断"),
          phase="llm",
        )
      except (json.JSONDecodeError, ValueError, KeyError):
        pass

    logger.warning("LLM 精判响应解析失败: %s", text[:100])
    return ReplyDecision(True, 5, "响应解析失败，默认回复", "llm")

  @staticmethod
  def _is_repetitive(content: str) -> bool:
    """检查是否为重复字符（如 "哈哈哈哈"、"666666"）"""
    if len(content) <= 1:
      return True
    unique_chars = set(content)
    return len(unique_chars) <= 2
