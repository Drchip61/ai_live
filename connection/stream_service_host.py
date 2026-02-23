"""
WebSocket 服务主机
提供弹幕输入和回复输出的 WebSocket 接口
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from streaming_studio import StreamingStudio, Comment, StreamerResponse
from streaming_studio.models import ResponseChunk


class StreamServiceHost:
  """
  WebSocket 服务主机
  管理客户端连接，区分输入者和输出者角色
  """

  def __init__(
    self,
    studio: StreamingStudio,
    host: str = "localhost",
    port: int = 8765
  ):
    """
    初始化服务主机

    Args:
      studio: 直播间实例
      host: 监听地址
      port: 监听端口
    """
    self.studio = studio
    self.host = host
    self.port = port

    # 客户端连接集合
    self._input_clients: Set[WebSocketServerProtocol] = set()
    self._output_clients: Set[WebSocketServerProtocol] = set()

    # 服务器实例
    self._server = None
    self._running = False

  @property
  def is_running(self) -> bool:
    """是否正在运行"""
    return self._running

  async def start(self) -> None:
    """启动 WebSocket 服务"""
    if self._running:
      return

    # 注册回复回调
    self.studio.on_response(self._on_response_sync)
    self.studio.on_response_chunk(self._on_chunk_sync)

    # 启动服务器
    self._server = await websockets.serve(
      self._handle_connection,
      self.host,
      self.port
    )
    self._running = True

    print(f"WebSocket 服务已启动: ws://{self.host}:{self.port}")

  async def stop(self) -> None:
    """停止 WebSocket 服务"""
    if not self._running:
      return

    self._running = False

    # 移除回调
    self.studio.remove_callback(self._on_response_sync)
    self.studio.remove_chunk_callback(self._on_chunk_sync)

    # 关闭所有客户端连接
    all_clients = self._input_clients | self._output_clients
    if all_clients:
      await asyncio.gather(
        *[client.close() for client in all_clients],
        return_exceptions=True
      )

    # 关闭服务器
    if self._server:
      self._server.close()
      await self._server.wait_closed()
      self._server = None

    self._input_clients.clear()
    self._output_clients.clear()

    print("WebSocket 服务已停止")

  async def _handle_connection(
    self,
    websocket: WebSocketServerProtocol,
    path: str
  ) -> None:
    """
    处理客户端连接

    Args:
      websocket: WebSocket 连接
      path: 请求路径
    """
    role = None

    try:
      # 等待角色注册消息
      try:
        register_msg = await asyncio.wait_for(
          websocket.recv(),
          timeout=10.0
        )
        data = json.loads(register_msg)

        if data.get("type") != "register":
          await websocket.send(json.dumps({
            "type": "error",
            "message": "需要先发送注册消息"
          }))
          return

        role = data.get("role")
        if role not in ("input", "output"):
          await websocket.send(json.dumps({
            "type": "error",
            "message": "无效的角色，必须是 'input' 或 'output'"
          }))
          return

      except asyncio.TimeoutError:
        await websocket.send(json.dumps({
          "type": "error",
          "message": "注册超时"
        }))
        return
      except json.JSONDecodeError:
        await websocket.send(json.dumps({
          "type": "error",
          "message": "无效的 JSON 格式"
        }))
        return

      # 注册客户端
      if role == "input":
        self._input_clients.add(websocket)
      else:
        self._output_clients.add(websocket)

      # 发送确认
      await websocket.send(json.dumps({
        "type": "registered",
        "role": role
      }))

      print(f"客户端已注册: role={role}, addr={websocket.remote_address}")

      # 根据角色处理消息
      if role == "input":
        await self._handle_input_client(websocket)
      else:
        await self._handle_output_client(websocket)

    except websockets.exceptions.ConnectionClosed:
      pass
    finally:
      # 清理连接
      self._input_clients.discard(websocket)
      self._output_clients.discard(websocket)
      print(f"客户端已断开: role={role}")

  async def _handle_input_client(
    self,
    websocket: WebSocketServerProtocol
  ) -> None:
    """
    处理输入客户端的消息

    Args:
      websocket: WebSocket 连接
    """
    async for message in websocket:
      try:
        data = json.loads(message)

        if data.get("type") != "comment":
          await websocket.send(json.dumps({
            "type": "error",
            "message": "未知的消息类型"
          }))
          continue

        # 创建弹幕
        comment = Comment(
          user_id=data.get("user_id", "anonymous"),
          nickname=data.get("nickname", "匿名用户"),
          content=data.get("content", "")
        )

        if not comment.content:
          await websocket.send(json.dumps({
            "type": "error",
            "message": "弹幕内容不能为空"
          }))
          continue

        # 发送到直播间
        self.studio.send_comment(comment)

        # 确认收到
        await websocket.send(json.dumps({
          "type": "comment_received",
          "comment_id": comment.id
        }))

      except json.JSONDecodeError:
        await websocket.send(json.dumps({
          "type": "error",
          "message": "无效的 JSON 格式"
        }))

  async def _handle_output_client(
    self,
    websocket: WebSocketServerProtocol
  ) -> None:
    """
    处理输出客户端
    保持连接直到断开

    Args:
      websocket: WebSocket 连接
    """
    # 输出客户端只需要保持连接，回复通过广播发送
    try:
      async for message in websocket:
        # 输出客户端可以发送心跳
        try:
          data = json.loads(message)
          if data.get("type") == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
        except json.JSONDecodeError:
          pass
    except websockets.exceptions.ConnectionClosed:
      pass

  def _on_response_sync(self, response: StreamerResponse) -> None:
    """
    同步的回复回调（在事件循环中异步执行广播）

    Args:
      response: 主播回复
    """
    # 创建异步任务来广播
    try:
      loop = asyncio.get_event_loop()
      if loop.is_running():
        asyncio.create_task(self._broadcast_response(response))
    except RuntimeError:
      pass

  async def _broadcast_response(self, response: StreamerResponse) -> None:
    """
    向所有输出客户端广播回复

    Args:
      response: 主播回复
    """
    if not self._output_clients:
      return

    message = json.dumps({
      "type": "response",
      "id": response.id,
      "content": response.content,
      "reply_to": list(response.reply_to),
      "timestamp": response.timestamp.isoformat()
    })

    # 并发发送给所有输出客户端
    await asyncio.gather(
      *[client.send(message) for client in self._output_clients],
      return_exceptions=True
    )

  def _on_chunk_sync(self, chunk: ResponseChunk) -> None:
    """
    同步的流式片段回调（在事件循环中异步执行广播）

    Args:
      chunk: 回复片段
    """
    try:
      loop = asyncio.get_event_loop()
      if loop.is_running():
        asyncio.create_task(self._broadcast_chunk(chunk))
    except RuntimeError:
      pass

  async def _broadcast_chunk(self, chunk: ResponseChunk) -> None:
    """
    向所有输出客户端广播回复片段

    Args:
      chunk: 回复片段
    """
    if not self._output_clients:
      return

    message = json.dumps({
      "type": "response_chunk",
      "response_id": chunk.response_id,
      "chunk": chunk.chunk,
      "accumulated": chunk.accumulated,
      "done": chunk.done,
    })

    await asyncio.gather(
      *[client.send(message) for client in self._output_clients],
      return_exceptions=True,
    )

  def get_stats(self) -> dict:
    """获取统计信息"""
    return {
      "is_running": self._running,
      "input_clients": len(self._input_clients),
      "output_clients": len(self._output_clients),
      "host": self.host,
      "port": self.port
    }
