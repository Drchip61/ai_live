"""
话题管理器编排器
统一管理话题表、弹幕分类、回复后分析和输出格式化
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

from .config import TopicManagerConfig
from .models import Topic, ContentAnalysisDelta, RhythmAnalysisDelta
from .table import TopicTable
from .classifier import single_classify, batch_classify
from .analyzer import post_reply_analysis
from .formatter import get_annotations, format_topic_context

if TYPE_CHECKING:
  from streaming_studio.database import CommentDatabase
  from streaming_studio.models import Comment

logger = logging.getLogger(__name__)


class TopicManager:
  """
  话题管理器

  职责：
  - 维护话题表（纯内存）
  - 分类弹幕到话题（单条 / 批量模式）
  - 回复后分析（内容分析 + 节奏分析）
  - 格式化话题上下文供 prompt 使用
  """

  def __init__(
    self,
    persona: str,
    database: "CommentDatabase",
    model: Optional[BaseChatModel] = None,
    config: Optional[TopicManagerConfig] = None,
  ):
    """
    初始化话题管理器

    Args:
      persona: 角色名称
      database: 弹幕数据库（用于取回弹幕内容）
      model: 小模型（延迟初始化）
      config: 配置
    """
    self._persona = persona
    self._database = database
    self._model = model
    self._config = config or TopicManagerConfig()

    # 话题表
    self._table = TopicTable(self._config)

    # 批量模式：弹幕收集队列
    self._pending_comments: list[tuple[str, str, str]] = []  # (id, user_id, content)
    self._batch_task: Optional[asyncio.Task] = None
    self._batch_event: asyncio.Event = asyncio.Event()

    # 分析状态
    self._last_analysis_time: Optional[datetime] = None
    self._suggested_timing: Optional[tuple[float, float]] = None

    # 后台任务引用（防止 GC 回收）
    self._background_tasks: set[asyncio.Task] = set()

    # 运行状态
    self._running = False

  def _get_model(self) -> BaseChatModel:
    """获取小模型（延迟初始化）"""
    if self._model is None:
      from langchain_wrapper.model_provider import ModelProvider
      self._model = ModelProvider.remote_small()
    return self._model

  @property
  def suggested_timing(self) -> Optional[tuple[float, float]]:
    """建议的下次等待时间 (min, max)"""
    return self._suggested_timing

  @property
  def table(self) -> TopicTable:
    """话题表（供外部只读访问）"""
    return self._table

  async def start(self) -> None:
    """
    启动话题管理器

    生成初始话题，启动批量分类循环（如果是批量模式）。
    """
    if self._running:
      return

    self._running = True

    # 初始化：生成开场话题（模板化，无模型调用）
    self._initialize()

    # 批量模式：启动收集循环
    if self._config.comment_mode == "batch":
      self._batch_task = asyncio.create_task(self._batch_loop())
      logger.info("话题管理器批量分类循环已启动")

    logger.info("话题管理器已启动 (角色: %s)", self._persona)

  async def stop(self) -> None:
    """停止话题管理器"""
    self._running = False

    if self._batch_task is not None:
      self._batch_task.cancel()
      try:
        await self._batch_task
      except asyncio.CancelledError:
        pass
      self._batch_task = None

    # 等待所有后台任务完成
    if self._background_tasks:
      await asyncio.gather(*self._background_tasks, return_exceptions=True)

    logger.info("话题管理器已停止")

  def _initialize(self) -> None:
    """生成模板初始话题（无模型调用）"""
    opening = Topic(
      topic_id="opening_greeting",
      title="开场打招呼",
      significance=self._config.initial_significance,
      topic_progress="刚开始直播，准备和观众打招呼",
      suggestion=f"可以用{self._persona}的风格问候观众，做自我介绍，聊聊今天的心情或计划",
    )
    self._table.add(opening)
    logger.info("初始话题已生成: opening_greeting")

  def on_comment(self, comment: "Comment") -> None:
    """
    接收弹幕（非阻塞）

    根据 comment_mode 选择单条或批量处理。

    Args:
      comment: 弹幕对象
    """
    if not self._running:
      return

    if self._config.comment_mode == "single":
      # 单条模式：立即异步分类
      task = asyncio.create_task(
        self._classify_single(comment.id, comment.user_id, comment.content)
      )
      self._background_tasks.add(task)
      task.add_done_callback(self._background_tasks.discard)
    else:
      # 批量模式：加入队列
      self._pending_comments.append(
        (comment.id, comment.user_id, comment.content)
      )
      # 达到阈值时立即触发
      if len(self._pending_comments) >= self._config.batch_size_threshold:
        self._batch_event.set()

  async def _classify_single(
    self,
    comment_id: str,
    user_id: str,
    content: str,
  ) -> None:
    """单条分类并关联"""
    try:
      topic_id = await single_classify(
        content, comment_id, self._table, self._get_model(),
      )
      if topic_id:
        self._table.add_comment_to_topic(topic_id, comment_id, user_id)
    except Exception:
      logger.exception("单条分类错误")

  async def _batch_loop(self) -> None:
    """批量分类循环"""
    while self._running:
      try:
        # 等待达到阈值或超时
        try:
          await asyncio.wait_for(
            self._batch_event.wait(),
            timeout=self._config.batch_wait_seconds,
          )
        except asyncio.TimeoutError:
          pass

        self._batch_event.clear()

        # 取出待分类弹幕
        if not self._pending_comments:
          continue

        batch = list(self._pending_comments)
        self._pending_comments.clear()

        # 批量分类
        results = await batch_classify(
          batch, self._table, self._get_model(),
        )

        # 关联弹幕到话题
        cid_to_uid = {cid: uid for cid, uid, _ in batch}
        for cid, topic_id in results.items():
          uid = cid_to_uid.get(cid, "")
          self._table.add_comment_to_topic(topic_id, cid, uid)

        if results:
          logger.debug("批量分类完成: %d/%d 条弹幕", len(results), len(batch))

      except asyncio.CancelledError:
        break
      except Exception:
        logger.exception("批量分类循环错误")
        await asyncio.sleep(1)

  def format_context(
    self,
    old_comments: list["Comment"],
    new_comments: list["Comment"],
  ) -> str:
    """
    格式化话题上下文（无模型请求）

    Args:
      old_comments: 上次回复前的弹幕
      new_comments: 上次回复后的新弹幕

    Returns:
      格式化的话题上下文文本
    """
    new_ids = [c.id for c in new_comments]
    return format_topic_context(
      self._table, new_ids, self._database, self._config,
    )

  def get_comment_annotations(self) -> dict[str, str]:
    """
    获取弹幕 → 话题的标注映射

    Returns:
      dict[comment_id, topic_id]
    """
    return get_annotations(self._table)

  async def post_reply(
    self,
    prompt: str,
    response: str,
    comments: list["Comment"],
  ) -> None:
    """
    回复后处理（fire-and-forget）

    包含：significance 衰减 + 按需分析。

    Args:
      prompt: 发给模型的 prompt
      response: 主播回复
      comments: 本轮相关弹幕
    """
    try:
      # 1. significance 衰减（无 LLM 调用）
      self._table.decay_all(self._config.significance_decay)

      # boost 最近弹幕提及的话题
      boosted = set()
      for comment in comments:
        topic = self._table.get_by_comment(comment.id)
        if topic and topic.topic_id not in boosted:
          self._table.boost(topic.topic_id, self._config.significance_boost)
          boosted.add(topic.topic_id)

      # 清理低 significance 话题
      self._table.cleanup(self._config.significance_threshold)

      # 2. 按需分析
      should_analyze = self._should_run_analysis(len(comments))
      if not should_analyze:
        return

      # 格式化弹幕文本
      comments_text = "\n".join(
        f"- {c.nickname}: {c.content}" for c in comments
      )

      # 并行分析
      content_delta, rhythm_delta = await post_reply_analysis(
        self._table, comments_text, response,
        self._get_model(), self._config,
      )

      # 统一应用 delta
      self._apply_content_delta(content_delta)
      self._apply_rhythm_delta(rhythm_delta)

      self._last_analysis_time = datetime.now()

    except Exception:
      logger.exception("回复后分析失败")

  def _should_run_analysis(self, comment_count: int) -> bool:
    """判断是否需要执行完整分析"""
    if comment_count < self._config.min_comments_for_analysis:
      logger.debug("弹幕不足 %d 条，跳过分析", self._config.min_comments_for_analysis)
      return False

    if self._last_analysis_time is not None:
      elapsed = (datetime.now() - self._last_analysis_time).total_seconds()
      if elapsed < self._config.min_analysis_interval_seconds:
        logger.debug("距上次分析仅 %.1f 秒，跳过", elapsed)
        return False

    return True

  def _apply_content_delta(self, delta: ContentAnalysisDelta) -> None:
    """应用内容分析 delta"""
    # 更新进度
    for topic_id, progress in delta.progress_updates.items():
      self._table.update(topic_id, topic_progress=progress)

    # 创建新话题
    for topic in delta.new_topics:
      self._table.add(topic)
      logger.info("新话题: %s - %s", topic.topic_id, topic.topic_progress)

    # 更新建议
    for topic_id, suggestion in delta.suggestion_updates.items():
      self._table.update(topic_id, suggestion=suggestion)

  def _apply_rhythm_delta(self, delta: RhythmAnalysisDelta) -> None:
    """应用节奏分析 delta"""
    # 标记过期话题
    for topic_id in delta.stale_topic_ids:
      self._table.update(topic_id, stale=True)

    # 更新动态等待时间
    if self._config.enable_dynamic_timing and delta.suggested_timing:
      self._suggested_timing = delta.suggested_timing
      logger.debug(
        "动态等待时间更新: %.1f ~ %.1f 秒",
        delta.suggested_timing[0], delta.suggested_timing[1],
      )

  def debug_state(self) -> dict:
    """
    获取调试状态快照

    Returns:
      包含话题管理器当前状态的字典
    """
    topics = self._table.get_all()
    return {
      "running": self._running,
      "comment_mode": self._config.comment_mode,
      "topic_count": self._table.count(),
      "topics": [
        {
          "topic_id": t.topic_id,
          "title": t.title,
          "significance": t.significance,
          "progress": t.topic_progress,
          "suggestion": t.suggestion,
          "stale": t.stale,
          "comment_count": len(t.comment_ids),
          "user_count": len(t.user_ids),
          "created_at": t.created_at.strftime("%H:%M:%S"),
          "updated_at": t.updated_at.strftime("%H:%M:%S"),
        }
        for t in sorted(topics, key=lambda x: x.significance, reverse=True)
      ],
      "pending_comments": len(self._pending_comments),
      "suggested_timing": self._suggested_timing,
      "last_analysis_time": (
        self._last_analysis_time.strftime("%H:%M:%S")
        if self._last_analysis_time else None
      ),
      "background_tasks": len(self._background_tasks),
    }
