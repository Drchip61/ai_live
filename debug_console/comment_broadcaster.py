"""
跨客户端广播器
在 NiceGUI 层面同步弹幕和回复显示，不涉及 studio 及以下层级。
多个 NiceGUI 客户端各自注册显示回调，任何来源的事件通过 broadcast() 推送给所有客户端。
"""

import logging
from collections.abc import Callable
from typing import Any

from streaming_studio.models import Comment, StreamerResponse, ResponseChunk

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
    for cb in list(self._callbacks):
      try:
        cb(comment)
      except Exception as e:
        logger.debug("弹幕广播回调错误: %s", e)

  def register(self, callback: Callable[[Comment], None]) -> None:
    self._callbacks.append(callback)

  def unregister(self, callback: Callable[[Comment], None]) -> None:
    if callback in self._callbacks:
      self._callbacks.remove(callback)

  @property
  def client_count(self) -> int:
    return len(self._callbacks)


class ResponseBroadcaster:
  """
  回复显示广播器

  将 studio 的 on_response / on_response_chunk 事件广播给所有 NiceGUI 客户端。
  每个客户端通过 page_client.safe_invoke() 接收，确保 UI 更新发送到正确的浏览器。
  """

  def __init__(self) -> None:
    self._chunk_callbacks: list[Callable[[ResponseChunk], None]] = []
    self._response_callbacks: list[Callable[[StreamerResponse], None]] = []
    self._pre_response_callbacks: list[Callable[[list, list], None]] = []

  def broadcast_chunk(self, chunk: ResponseChunk) -> None:
    for cb in list(self._chunk_callbacks):
      try:
        cb(chunk)
      except Exception as e:
        logger.debug("chunk 广播回调错误: %s", e)

  def broadcast_response(self, response: StreamerResponse) -> None:
    for cb in list(self._response_callbacks):
      try:
        cb(response)
      except Exception as e:
        logger.debug("response 广播回调错误: %s", e)

  def broadcast_pre_response(self, old_comments: list, new_comments: list) -> None:
    for cb in list(self._pre_response_callbacks):
      try:
        cb(old_comments, new_comments)
      except Exception as e:
        logger.debug("pre_response 广播回调错误: %s", e)

  def register_chunk(self, callback: Callable[[ResponseChunk], None]) -> None:
    self._chunk_callbacks.append(callback)

  def unregister_chunk(self, callback: Callable[[ResponseChunk], None]) -> None:
    if callback in self._chunk_callbacks:
      self._chunk_callbacks.remove(callback)

  def register_response(self, callback: Callable[[StreamerResponse], None]) -> None:
    self._response_callbacks.append(callback)

  def unregister_response(self, callback: Callable[[StreamerResponse], None]) -> None:
    if callback in self._response_callbacks:
      self._response_callbacks.remove(callback)

  def register_pre_response(self, callback: Callable[[list, list], None]) -> None:
    self._pre_response_callbacks.append(callback)

  def unregister_pre_response(self, callback: Callable[[list, list], None]) -> None:
    if callback in self._pre_response_callbacks:
      self._pre_response_callbacks.remove(callback)

  @property
  def client_count(self) -> int:
    return len(self._chunk_callbacks)
