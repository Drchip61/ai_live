"""
自动观众模式引擎
使用小模型自动生成虚拟观众弹幕
"""

import asyncio
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from coolname import generate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_wrapper.model_provider import ModelProvider
from streaming_studio import StreamingStudio, Comment
from streaming_studio.models import StreamerResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoViewerConfig:
  """
  自动观众配置

  Attributes:
    min_interval: 最小生成间隔（秒）
    max_interval: 最大生成间隔（秒）
    max_responses_context: 主播回复上下文数量（最近 N 条）
    viewer_pool_size: 固定观众池大小
  """
  min_interval: float = 10.0
  max_interval: float = 15.0
  max_responses_context: int = 5
  viewer_pool_size: int = 6
  viewer_rotate_interval: float = 60.0
  """观众轮换间隔（秒），每隔这么久随机换掉 1-2 个观众"""


def _load_prompt() -> str:
  """加载虚拟观众提示词"""
  path = Path(__file__).parent.parent / "prompts" / "auto_viewer.txt"
  return path.read_text(encoding="utf-8")


class AutoViewer:
  """
  自动观众引擎

  监听主播回复，定时调用小模型生成虚拟观众弹幕。
  """

  def __init__(
    self,
    studio: StreamingStudio,
    config: AutoViewerConfig = AutoViewerConfig(),
  ):
    """
    初始化自动观众引擎

    Args:
      studio: 直播间实例
      config: 配置对象
    """
    self.studio = studio
    self.config = config
    self._model = ModelProvider.remote_small()
    self._prompt_template = _load_prompt()
    self._recent_responses: list[str] = []
    self._running = False
    self._task: Optional[asyncio.Task] = None
    self._response_cb: Optional[Callable] = None
    self._comment_callbacks: list[Callable[[Comment], None]] = []

    # 固定观众池
    self._viewer_pool: list[tuple[str, str]] = [
      _random_identity() for _ in range(config.viewer_pool_size)
    ]
    self._time_since_rotate: float = 0.0

  @property
  def is_running(self) -> bool:
    """是否正在运行"""
    return self._running

  def on_comment(self, callback: Callable[[Comment], None]) -> None:
    """注册弹幕生成回调（供 UI 显示用）"""
    self._comment_callbacks.append(callback)

  def remove_comment_callback(self, callback: Callable[[Comment], None]) -> None:
    """移除弹幕生成回调"""
    if callback in self._comment_callbacks:
      self._comment_callbacks.remove(callback)

  async def start(self) -> None:
    """启动自动观众：注册回调 + 启动生成循环"""
    if self._running:
      return

    self._running = True
    self._response_cb = self._on_response
    self.studio.on_response(self._response_cb)
    self._task = asyncio.create_task(self._generation_loop())
    logger.info("自动观众已启动")

  async def stop(self) -> None:
    """停止自动观众：移除回调 + 取消任务"""
    self._running = False

    if self._response_cb is not None:
      self.studio.remove_callback(self._response_cb)
      self._response_cb = None

    if self._task is not None:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
      self._task = None

    logger.info("自动观众已停止")

  def _on_response(self, response: StreamerResponse) -> None:
    """
    主播回复回调：追踪最近回复内容

    Args:
      response: 主播回复
    """
    self._recent_responses.append(response.content)
    max_ctx = self.config.max_responses_context
    if len(self._recent_responses) > max_ctx:
      self._recent_responses = self._recent_responses[-max_ctx:]

  async def _generation_loop(self) -> None:
    """生成循环：随机间隔生成弹幕批次"""
    while self._running:
      try:
        interval = random.uniform(
          self.config.min_interval,
          self.config.max_interval,
        )
        await asyncio.sleep(interval)

        if not self._running:
          break

        # 定期轮换观众
        self._time_since_rotate += interval
        if self._time_since_rotate >= self.config.viewer_rotate_interval:
          self._rotate_viewers()
          self._time_since_rotate = 0.0

        if self._recent_responses:
          await self._generate_and_send()
      except asyncio.CancelledError:
        break
      except Exception as e:
        logger.error("自动观众生成错误: %s", e)
        await asyncio.sleep(2)

  def _rotate_viewers(self) -> None:
    """随机换掉 1-2 个观众，模拟观众进出"""
    count = random.randint(1, 2)
    indices = random.sample(range(len(self._viewer_pool)), min(count, len(self._viewer_pool)))
    for i in indices:
      old_name = self._viewer_pool[i][1]
      self._viewer_pool[i] = _random_identity()
      logger.info("观众轮换: %s → %s", old_name, self._viewer_pool[i][1])

  async def _generate_and_send(self) -> None:
    """调用小模型生成弹幕并发送"""
    context = self._build_context()
    messages = [
      SystemMessage(content=self._prompt_template),
      HumanMessage(content=f"主播最近发言：\n{context}\n\n请生成观众弹幕："),
    ]

    try:
      result = await self._model.ainvoke(messages)
      text = result.content if hasattr(result, "content") else str(result)

      lines = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip()
      ]

      for line in lines[:4]:
        user_id, nickname = random.choice(self._viewer_pool)
        comment = Comment(
          user_id=user_id,
          nickname=nickname,
          content=line,
        )
        self.studio.send_comment(comment)
        for cb in self._comment_callbacks:
          try:
            cb(comment)
          except Exception as e:
            logger.debug("弹幕回调错误: %s", e)
        await asyncio.sleep(random.uniform(0.3, 1.0))

    except Exception as e:
      logger.error("自动观众 LLM 调用失败: %s", e)

  def _build_context(self) -> str:
    """
    构建主播回复上下文

    Returns:
      格式化的最近回复文本
    """
    return "\n".join(
      f"- {resp}" for resp in self._recent_responses
    )


def _random_identity() -> tuple[str, str]:
  """
  生成随机虚拟观众身份

  Returns:
    (user_id, nickname)
  """
  words = generate(2)
  user_id = "_".join(w.lower() for w in words)
  nickname = " ".join(w.capitalize() for w in words)
  return user_id, nickname
