"""
DanmakuPushHost 回归测试

运行方式:
  python test_danmaku_push_host.py
"""

import asyncio
import socket
import sys
from pathlib import Path

import aiohttp

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from connection.danmaku_push_host import DanmakuPushHost, parse_snapshot_payload
from streaming_studio.models import EventType


def _free_port() -> int:
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("127.0.0.1", 0))
  port = sock.getsockname()[1]
  sock.close()
  return port


class _DummyStudio:
  def __init__(self):
    self.comments = []

  def send_comment(self, comment):
    self.comments.append(comment)


def test_parse_snapshot_payload_keeps_existing_envelope_shape():
  payload = {
    "danmakus": [
      {
        "content": "你好",
        "user_id": "u1",
        "nickname": "弹幕A",
        "timestamp": 1710000000.0,
      }
    ],
    "super_chats": [
      {
        "message": "加油",
        "uid": "u2",
        "nickname": "SC大佬",
        "price": 30,
        "timestamp": 1710000001.0,
      }
    ],
    "server_time": 1710000002.0,
  }
  items, server_time = parse_snapshot_payload(payload)
  assert len(items) == 2
  assert server_time == 1710000002.0
  assert items[0].event_type == "danmaku"
  assert items[0].content == "你好"
  assert items[1].event_type == "super_chat"
  assert items[1].content == "加油"
  print("  [PASS] snapshot envelope 解析正确")


async def test_push_host_accepts_snapshot_and_dedupes_retries():
  studio = _DummyStudio()
  port = _free_port()
  host = DanmakuPushHost(
    studio,
    host="127.0.0.1",
    port=port,
    path="/snapshot",
    dedupe_maxlen=20,
  )
  await host.start()
  payload = {
    "danmakus": [
      {
        "content": "第一条",
        "user_id": "u1",
        "nickname": "弹幕A",
        "timestamp": 1710000100.0,
      },
      {
        "content": "第二条",
        "user_id": "u1",
        "nickname": "弹幕A",
        "timestamp": 1710000101.0,
      },
    ],
    "super_chats": [
      {
        "message": "冲冲冲",
        "uid": "u2",
        "nickname": "SC大佬",
        "price": 50,
        "timestamp": 1710000102.0,
      }
    ],
    "server_time": 1710000103.0,
  }
  try:
    async with aiohttp.ClientSession() as session:
      async with session.post(f"http://127.0.0.1:{port}/snapshot", json=payload) as resp1:
        ack1 = await resp1.json()
      async with session.post(f"http://127.0.0.1:{port}/snapshot", json=payload) as resp2:
        ack2 = await resp2.json()
  finally:
    await host.stop()

  assert ack1["ok"] is True
  assert ack1["accepted"] == 3
  assert ack1["duplicates"] == 0
  assert ack1["accepted_counts"]["danmaku"] == 2
  assert ack1["accepted_counts"]["super_chat"] == 1

  assert ack2["ok"] is True
  assert ack2["accepted"] == 0
  assert ack2["duplicates"] == 3
  assert len(studio.comments) == 3
  assert studio.comments[0].receive_seq == 0
  assert studio.comments[0].event_type == EventType.DANMAKU
  assert studio.comments[2].event_type == EventType.SUPER_CHAT
  print("  [PASS] push host ACK/去重/入缓冲计数一致")


def main():
  tests = [
    test_parse_snapshot_payload_keeps_existing_envelope_shape,
    test_push_host_accepts_snapshot_and_dedupes_retries,
  ]
  failed = 0
  for test_fn in tests:
    try:
      if asyncio.iscoroutinefunction(test_fn):
        asyncio.run(test_fn())
      else:
        test_fn()
    except AssertionError as e:
      failed += 1
      print(f"  [FAIL] {test_fn.__name__}: {e}")
    except Exception as e:
      failed += 1
      print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
