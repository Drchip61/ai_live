"""
弹幕广播器
在 debug_console（客户端播放器）层面同步弹幕显示，不涉及 studio 及以下层级。
多个 NiceGUI 客户端各自注册显示回调，任何来源的弹幕通过 broadcast() 推送给所有客户端。
"""

import logging
from collections.abc import Callable

from streaming_studio.models import Comment

logger = logging.getLogger(__name__)


class CommentBroadcaster:
  """
  弹幕显示广播器

  职责：维护跨客户端的弹幕显示回调列表。
  弹幕的业务处理（送入 studio 缓冲区）仍由调用方负责，
  本类只负责"让所有连接的客户端都能看到这条弹幕"。
  """

  def __init__(self) -> None:
    self._callbacks: list[Callable[[Comment], None]] = []

  def broadcast(self, comment: Comment) -> None:
    """
    广播一条弹幕到所有已注册的客户端

    Args:
      comment: 弹幕对象
    """
    for cb in list(self._callbacks):
      try:
        cb(comment)
      except Exception as e:
        logger.debug("弹幕广播回调错误: %s", e)

  def register(self, callback: Callable[[Comment], None]) -> None:
    """注册显示回调（每个客户端连接时调用）"""
    self._callbacks.append(callback)

  def unregister(self, callback: Callable[[Comment], None]) -> None:
    """移除显示回调（客户端断开时调用）"""
    if callback in self._callbacks:
      self._callbacks.remove(callback)

  @property
  def client_count(self) -> int:
    """当前已注册的客户端数量"""
    return len(self._callbacks)
