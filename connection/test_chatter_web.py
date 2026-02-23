"""
WebSocket 测试客户端
用于测试 WebSocket 服务
"""

import asyncio
import json
import sys
from pathlib import Path

import websockets


class TestChatterWeb:
  """WebSocket 测试客户端"""

  def __init__(self, uri: str = "ws://localhost:8765"):
    self.uri = uri
    self.user_id: str = ""
    self.nickname: str = ""

  async def run(self) -> None:
    """运行测试客户端"""
    print("=" * 50)
    print("WebSocket 测试客户端")
    print("=" * 50)
    print()

    # 获取用户信息
    self.user_id = input("请输入你的用户ID: ").strip() or "test_user"
    self.nickname = input("请输入你的昵称: ").strip() or "测试用户"
    print()

    # 获取服务器地址
    custom_uri = input(f"服务器地址 (默认 {self.uri}): ").strip()
    if custom_uri:
      self.uri = custom_uri
    print()

    print(f"正在连接到 {self.uri}...")
    print()

    try:
      # 同时建立输入和输出连接
      async with websockets.connect(self.uri) as input_ws, \
                 websockets.connect(self.uri) as output_ws:

        # 注册输入角色
        await input_ws.send(json.dumps({
          "type": "register",
          "role": "input"
        }))
        input_response = await input_ws.recv()
        input_data = json.loads(input_response)
        if input_data.get("type") != "registered":
          print(f"输入注册失败: {input_data}")
          return
        print("输入连接已建立")

        # 注册输出角色
        await output_ws.send(json.dumps({
          "type": "register",
          "role": "output"
        }))
        output_response = await output_ws.recv()
        output_data = json.loads(output_response)
        if output_data.get("type") != "registered":
          print(f"输出注册失败: {output_data}")
          return
        print("输出连接已建立")

        print()
        print("=" * 50)
        print("连接成功！输入弹幕后按回车发送")
        print("输入 /quit 退出")
        print("=" * 50)
        print()

        # 启动接收任务
        receive_task = asyncio.create_task(
          self._receive_responses(output_ws)
        )

        # 启动发送任务
        send_task = asyncio.create_task(
          self._send_comments(input_ws)
        )

        # 等待任一任务完成
        done, pending = await asyncio.wait(
          [receive_task, send_task],
          return_when=asyncio.FIRST_COMPLETED
        )

        # 取消未完成的任务
        for task in pending:
          task.cancel()
          try:
            await task
          except asyncio.CancelledError:
            pass

    except websockets.exceptions.ConnectionClosed as e:
      print(f"连接已关闭: {e}")
    except ConnectionRefusedError:
      print(f"无法连接到服务器 {self.uri}")
      print("请确保服务器已启动")
    except Exception as e:
      print(f"错误: {e}")

  async def _send_comments(self, websocket) -> None:
    """发送弹幕"""
    loop = asyncio.get_event_loop()

    while True:
      try:
        # 在单独线程中读取输入
        user_input = await loop.run_in_executor(
          None,
          lambda: input(f"[{self.nickname}] > ")
        )
      except EOFError:
        break

      user_input = user_input.strip()
      if not user_input:
        continue

      if user_input == "/quit":
        print("再见！")
        break

      # 发送弹幕
      await websocket.send(json.dumps({
        "type": "comment",
        "user_id": self.user_id,
        "nickname": self.nickname,
        "content": user_input
      }))

      # 等待确认
      try:
        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        data = json.loads(response)
        if data.get("type") == "error":
          print(f"发送失败: {data.get('message')}")
      except asyncio.TimeoutError:
        print("发送确认超时")

  async def _receive_responses(self, websocket) -> None:
    """接收回复"""
    try:
      async for message in websocket:
        try:
          data = json.loads(message)
          if data.get("type") == "response":
            print()
            print(f"[主播] {data.get('content')}")
            print()
        except json.JSONDecodeError:
          pass
    except websockets.exceptions.ConnectionClosed:
      pass


async def main():
  client = TestChatterWeb()
  try:
    await client.run()
  except KeyboardInterrupt:
    print("\n已中断")


if __name__ == "__main__":
  asyncio.run(main())
