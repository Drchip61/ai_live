"""
聚焦会话管理器
让主播在遇到高质量弹幕或精彩画面时进入"聚焦模式"，
围绕核心话题连续展开多轮深入互动。

两种会话类型：
  CommentSession — 高质量弹幕触发，后续 2-3 轮围绕该话题展开
  VideoSession   — 连续精彩画面 + 无弹幕，主播深入评论视频内容
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .config import SessionConfig
from .models import Comment, EventType

logger = logging.getLogger(__name__)

_NOISE_PATTERN = re.compile(
  r"^(哈+|6+|[?？!！.。~～、，]+|草+|好家伙|啊+|呜+|嗯+|ww+|hhh+|lol+|emm+|"
  r"nb|tql|xswl|yyds|awsl|dd|ddd+|[Oo0]+|233+|7777*|牛|强|绝|顶|冲|来了|"
  r"你好|hello|hi|嗨|晚上好|早上好|下午好|主播好|"
  r"[👍👏🔥❤️💯😂🤣😭😍]+)$",
  re.IGNORECASE,
)

_OPINION_MARKERS = re.compile(
  r"觉得|认为|推荐|为什么|怎么[样办]|好不好|有没有|"
  r"是不是|应该|其实|感觉|建议|对比|区别|选择|分析|"
  r"讲讲|说说|聊聊|介绍|评价|看法"
)

_EXPRESSION_TAG_RE = re.compile(r"#\[[^\]]*\]\[[^\]]*\]\s*")


class SessionType(str, Enum):
  COMMENT = "comment"
  VIDEO = "video"


@dataclass
class FocusSession:
  """聚焦会话状态"""
  session_type: SessionType
  anchor_text: str
  """触发话题的核心文本（弹幕内容 / 场景描述）"""

  anchor_comment: Optional[Comment] = None
  """触发弹幕（仅 CommentSession）"""

  round_count: int = 0
  """已完成的回复轮数"""

  max_rounds: int = 3
  """最大轮数"""

  history: list[str] = field(default_factory=list)
  """每轮回复摘要"""

  created_at: float = field(default_factory=time.monotonic)
  last_active_at: float = field(default_factory=time.monotonic)
  """最近一次有相关弹幕或回复的时间"""

  stale_rounds: int = 0
  """连续无相关弹幕的轮数（用于判断 session 是否该结束）"""

  interrupted: bool = False
  """VideoSession 是否被弹幕打断（临时让出优先级）"""


class SessionManager:
  """
  聚焦会话管理器

  在主循环的每一轮中被调用，负责：
  - 判断是否应开启新 session（高质量弹幕 / 精彩画面）
  - 跟踪 session 生命周期（轮次递增、相关性检测、自动结束）
  - 生成 prompt 上下文（注入聚焦指令）
  - 处理 VideoSession 被弹幕打断的优先级切换
  """

  def __init__(self, config: Optional[SessionConfig] = None):
    self._config = config or SessionConfig()
    self._session: Optional[FocusSession] = None

  @property
  def is_active(self) -> bool:
    return self._session is not None

  @property
  def session(self) -> Optional[FocusSession]:
    return self._session

  @property
  def session_type(self) -> Optional[SessionType]:
    return self._session.session_type if self._session else None

  # ── 高质量弹幕判定 ──

  def _is_quality_comment(self, comment: Comment) -> bool:
    """
    判断单条弹幕是否为"高质量"，值得开启聚焦会话。

    极简规则：SC/priority 直接通过，普通弹幕非噪声且 >= 3字即触发。
    """
    if comment.event_type == EventType.SUPER_CHAT:
      return True
    if comment.priority:
      return True

    if comment.event_type != EventType.DANMAKU:
      return False

    content = comment.content.strip()
    if not content:
      return False

    if _NOISE_PATTERN.match(content):
      return False

    if len(set(content)) <= 2 and len(content) > 1:
      return False

    return len(content) >= self._config.long_content_length

  def find_quality_comment(self, comments: list[Comment]) -> Optional[Comment]:
    """从弹幕列表中找到第一条高质量弹幕（按优先级排序后）"""
    if not comments:
      return None
    for c in comments:
      if self._is_quality_comment(c):
        return c
    return None

  # ── Session 生命周期 ──

  def evaluate_new_session(
    self,
    new_comments: list[Comment],
    old_comments: list[Comment],
    has_scene_change: bool = False,
    silence_seconds: float = 0,
  ) -> bool:
    """
    评估是否应开启新 session。

    如果已有活跃 session，只在以下情况切换：
    - 当前是 VideoSession 且出现高质量弹幕 → 切换为 CommentSession

    Returns:
      True 表示开启/切换了新 session
    """
    if not self._config.enabled:
      return False

    all_new = new_comments
    quality = self.find_quality_comment(all_new)

    # 已有 session 的情况
    if self._session is not None:
      # VideoSession 被高质量弹幕打断 → 切换为 CommentSession
      if (
        self._session.session_type == SessionType.VIDEO
        and quality is not None
      ):
        logger.info(
          "VideoSession 被高质量弹幕打断，切换为 CommentSession: %s",
          quality.content[:30],
        )
        self._start_comment_session(quality)
        return True
      # 已有 CommentSession 时不切换（让当前 session 自然结束）
      return False

    # 无活跃 session → 尝试开启新的
    if quality is not None:
      self._start_comment_session(quality)
      return True

    # 尝试开启 VideoSession：需要画面变化 + 无弹幕 + 足够沉默
    if (
      has_scene_change
      and not new_comments
      and not old_comments
      and silence_seconds >= self._config.video_session_silence_threshold
    ):
      self._start_video_session()
      return True

    return False

  def _start_comment_session(self, trigger: Comment) -> None:
    self._session = FocusSession(
      session_type=SessionType.COMMENT,
      anchor_text=trigger.content.strip(),
      anchor_comment=trigger,
      max_rounds=self._config.comment_session_max_rounds,
    )
    logger.info(
      "CommentSession 开启: 「%s」 (by %s), max_rounds=%d",
      trigger.content[:40], trigger.nickname,
      self._config.comment_session_max_rounds,
    )
    print(f"[聚焦会话] CommentSession 开启: 「{trigger.content[:40]}」 by {trigger.nickname}")

  def _start_video_session(self) -> None:
    self._session = FocusSession(
      session_type=SessionType.VIDEO,
      anchor_text="视频画面",
      max_rounds=self._config.video_session_max_rounds,
    )
    logger.info(
      "VideoSession 开启, max_rounds=%d",
      self._config.video_session_max_rounds,
    )
    print("[聚焦会话] VideoSession 开启: 进入看视频模式")

  def update_after_response(
    self,
    response_text: str,
    new_comments: list[Comment],
  ) -> None:
    """
    回复完成后更新 session 状态。

    递增轮次、追加历史、检测相关性、判断是否结束。
    """
    if self._session is None:
      return

    s = self._session
    s.round_count += 1

    # 记录回复摘要（去掉表情标签，截取前 80 字）
    cleaned = _EXPRESSION_TAG_RE.sub("", response_text).strip()
    sep_idx = cleaned.find(" / ")
    if sep_idx >= 0:
      cleaned = cleaned[:sep_idx].strip()
    s.history.append(cleaned[:80])
    if len(s.history) > 5:
      s.history = s.history[-5:]

    # 相关性检测
    if s.session_type == SessionType.COMMENT:
      has_relevant = self._has_relevant_comment(new_comments, s.anchor_text)
      if has_relevant:
        s.stale_rounds = 0
        s.last_active_at = time.monotonic()
      else:
        s.stale_rounds += 1
    else:
      # VideoSession：只要还在看视频就活跃
      s.last_active_at = time.monotonic()
      s.stale_rounds = 0

    # 检查是否应结束
    if self._should_end():
      self.end_session()

  def _has_relevant_comment(
    self,
    comments: list[Comment],
    anchor_text: str,
  ) -> bool:
    """
    检测弹幕列表中是否有与 anchor 话题相关的内容。

    简单关键词重叠法：anchor 中的关键词（>=2字）在弹幕中出现即视为相关。
    """
    if not comments:
      return False

    anchor_chars = set(anchor_text)
    # 提取 anchor 中连续 2+ 字的片段作为关键词
    keywords = set()
    for i in range(len(anchor_text) - 1):
      keywords.add(anchor_text[i:i+2])

    for c in comments:
      if c.event_type != EventType.DANMAKU:
        continue
      content = c.content.strip()
      if not content or len(content) < 2:
        continue
      # 噪声排除
      if _NOISE_PATTERN.match(content):
        continue
      # 关键词重叠
      overlap = sum(1 for kw in keywords if kw in content)
      if overlap >= 2:
        return True
      # 字符重叠率
      common = len(set(content) & anchor_chars)
      if common >= 3 and common / len(set(content)) > 0.3:
        return True

    return False

  def _should_end(self) -> bool:
    """判断当前 session 是否应结束"""
    if self._session is None:
      return False

    s = self._session

    # 达到最大轮数
    if s.round_count >= s.max_rounds:
      logger.info("Session 达到最大轮数 (%d/%d)", s.round_count, s.max_rounds)
      return True

    # 连续无相关弹幕
    if s.stale_rounds >= self._config.stale_rounds_to_end:
      logger.info("Session 连续 %d 轮无相关弹幕", s.stale_rounds)
      return True

    # 超时
    elapsed = time.monotonic() - s.last_active_at
    if elapsed > self._config.relevance_timeout:
      logger.info("Session 超时 (%.0fs 无活动)", elapsed)
      return True

    return False

  def end_session(self) -> None:
    """结束当前 session"""
    if self._session is not None:
      logger.info(
        "Session 结束: type=%s, rounds=%d/%d",
        self._session.session_type.value,
        self._session.round_count,
        self._session.max_rounds,
      )
      print(
        f"[聚焦会话] Session 结束: {self._session.session_type.value}, "
        f"轮次 {self._session.round_count}/{self._session.max_rounds}"
      )
    self._session = None

  def on_comment_arrived(self, new_comments: list[Comment]) -> None:
    """
    VideoSession 专用：弹幕到达时处理打断逻辑。

    高质量弹幕 → 切换为 CommentSession（在 evaluate_new_session 中处理）
    普通弹幕 → 标记 interrupted，让 prompt 优先弹幕
    """
    if self._session is None:
      return
    if self._session.session_type != SessionType.VIDEO:
      return
    if not new_comments:
      self._session.interrupted = False
      return

    # 有弹幕到达：标记打断（但不结束 session，除非是高质量弹幕）
    has_real = any(
      c.event_type == EventType.DANMAKU and c.content.strip()
      for c in new_comments
    )
    self._session.interrupted = has_real

  # ── Prompt 生成 ──

  def to_prompt(self) -> str:
    """生成注入 LLM prompt 的 session 上下文，无 session 时返回空字符串"""
    if self._session is None:
      return ""

    s = self._session

    if s.session_type == SessionType.COMMENT:
      return self._comment_session_prompt(s)
    else:
      return self._video_session_prompt(s)

  def _comment_session_prompt(self, s: FocusSession) -> str:
    round_display = s.round_count + 1
    parts = [
      f"[聚焦话题] 当前正在深入讨论一个有趣的话题（第{round_display}/{s.max_rounds}轮）。",
      f"核心话题：「{s.anchor_text[:60]}」",
    ]
    if s.history:
      history_text = "；".join(s.history[-3:])
      parts.append(f"之前讨论：{history_text}")
    parts.append(
      "请围绕这个话题继续展开，可以追问细节、分享观点、挖掘更多有趣的角度。"
      "如果有其他弹幕，也可以自然地引回这个话题。"
    )
    return "\n".join(parts) + "\n\n"

  def _video_session_prompt(self, s: FocusSession) -> str:
    round_display = s.round_count + 1
    parts = [
      f"[看视频模式] 你正在认真观看视频内容（第{round_display}/{s.max_rounds}轮）。",
      "请深入评论当前画面，展开你的感受和想法，不要只用一句话概括。",
    ]
    if s.history:
      history_text = "；".join(s.history[-3:])
      parts.append(f"你之前说了：{history_text}")
      parts.append("请继续延伸，不要重复已经说过的内容。")

    if s.interrupted:
      parts.append(
        "注意：有弹幕到来了，请优先回复弹幕内容。"
        "但你可以把弹幕和正在看的视频内容联系起来。"
      )
    return "\n".join(parts) + "\n\n"

  # ── 调试 ──

  def debug_state(self) -> dict:
    if self._session is None:
      return {"active": False}

    s = self._session
    return {
      "active": True,
      "type": s.session_type.value,
      "anchor_text": s.anchor_text[:60],
      "anchor_user": s.anchor_comment.nickname if s.anchor_comment else None,
      "round": s.round_count,
      "max_rounds": s.max_rounds,
      "stale_rounds": s.stale_rounds,
      "interrupted": s.interrupted,
      "history": s.history[-3:],
      "age_seconds": round(time.monotonic() - s.created_at),
      "since_active_seconds": round(time.monotonic() - s.last_active_at),
    }
