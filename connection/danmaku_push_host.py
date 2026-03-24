"""
推送式弹幕接入 Host

复用 snapshot payload 的事件解析逻辑，供：
- run_remote.py 的 9100 push ingress
- RemoteSource 的旧轮询模式（截图仍可保留 pull）
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any, Optional

from aiohttp import web

from streaming_studio.models import Comment, EventType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteDanmakuItem:
  """上游 snapshot 事件条目的中间表示。"""

  content: str
  user_id: str = ""
  nickname: str = ""
  timestamp: float = 0.0
  event_type: str = "danmaku"
  gift_name: str = ""
  gift_num: int = 0
  price: float = 0.0
  guard_level: int = 0


def _as_float(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _as_int(value: Any, default: int = 0) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _pick_user_id(raw: dict[str, Any], fallback: str = "") -> str:
  for key in ("uid", "user_id", "mid", "id"):
    value = raw.get(key)
    if value not in (None, ""):
      return str(value)
  return fallback


def _pick_nickname(raw: dict[str, Any], fallback: str = "") -> str:
  for key in ("username", "nickname", "uname", "name"):
    value = raw.get(key)
    if value not in (None, ""):
      return str(value)
  return fallback


def parse_snapshot_payload(data: Any) -> tuple[list[RemoteDanmakuItem], float]:
  """解析上游 snapshot envelope。"""
  if not isinstance(data, dict):
    return [], 0.0

  server_time = _as_float(data.get("server_time"), 0.0)
  items: list[RemoteDanmakuItem] = []

  for raw in data.get("danmakus", []) or []:
    if not isinstance(raw, dict):
      continue
    content = str(raw.get("content", "") or "").strip()
    nickname = _pick_nickname(raw, "匿名")
    items.append(RemoteDanmakuItem(
      content=content,
      user_id=_pick_user_id(raw, nickname),
      nickname=nickname,
      timestamp=_as_float(raw.get("timestamp"), server_time),
      event_type="danmaku",
    ))

  for raw in data.get("notifications", []) or []:
    if not isinstance(raw, dict):
      continue
    nickname = _pick_nickname(raw, "观众")
    items.append(RemoteDanmakuItem(
      content=str(raw.get("content", "") or "").strip(),
      user_id=_pick_user_id(raw, nickname),
      nickname=nickname,
      timestamp=_as_float(raw.get("timestamp"), server_time),
      event_type="entry",
      guard_level=_as_int(raw.get("guard_level"), 0),
    ))

  for raw in data.get("gifts", []) or []:
    if not isinstance(raw, dict):
      continue
    nickname = _pick_nickname(raw, "观众")
    items.append(RemoteDanmakuItem(
      content="",
      user_id=_pick_user_id(raw, nickname),
      nickname=nickname,
      timestamp=_as_float(raw.get("timestamp"), server_time),
      event_type="gift",
      gift_name=str(raw.get("gift_name", "") or "").strip(),
      gift_num=_as_int(raw.get("gift_num"), _as_int(raw.get("num"), 0)),
      price=_as_float(raw.get("price"), 0.0),
    ))

  for raw in data.get("super_chats", []) or []:
    if not isinstance(raw, dict):
      continue
    nickname = _pick_nickname(raw, "观众")
    items.append(RemoteDanmakuItem(
      content=str(raw.get("content", raw.get("message", "")) or "").strip(),
      user_id=_pick_user_id(raw, nickname),
      nickname=nickname,
      timestamp=_as_float(raw.get("timestamp"), server_time),
      event_type="super_chat",
      price=_as_float(raw.get("price"), 0.0),
    ))

  for raw in data.get("guard_buys", []) or []:
    if not isinstance(raw, dict):
      continue
    nickname = _pick_nickname(raw, "观众")
    items.append(RemoteDanmakuItem(
      content="",
      user_id=_pick_user_id(raw, nickname),
      nickname=nickname,
      timestamp=_as_float(raw.get("timestamp"), server_time),
      event_type="guard_buy",
      guard_level=_as_int(raw.get("guard_level"), 1),
      gift_num=max(1, _as_int(raw.get("months"), _as_int(raw.get("gift_num"), _as_int(raw.get("num"), 1)))),
      price=_as_float(raw.get("price"), 0.0),
    ))

  return items, server_time


def item_fingerprint(item: RemoteDanmakuItem) -> tuple[Any, ...]:
  """请求重试/重复推送去重指纹。"""
  return (
    item.event_type,
    item.user_id,
    item.nickname,
    item.content,
    round(float(item.timestamp or 0.0), 3),
    item.gift_name,
    int(item.gift_num or 0),
    round(float(item.price or 0.0), 2),
    int(item.guard_level or 0),
  )


def remote_item_to_comment(item: RemoteDanmakuItem) -> Comment:
  """将中间表示转换为 Comment。"""
  try:
    event_type = EventType(item.event_type)
  except ValueError:
    event_type = EventType.DANMAKU

  ts = datetime.fromtimestamp(item.timestamp) if item.timestamp > 0 else datetime.now()
  raw_uid = item.user_id
  uid = raw_uid if raw_uid and raw_uid != "0" else (item.nickname or "anonymous")

  return Comment(
    user_id=uid,
    nickname=item.nickname or uid,
    content=item.content,
    timestamp=ts,
    event_type=event_type,
    gift_name=item.gift_name,
    gift_num=item.gift_num,
    price=item.price,
    guard_level=item.guard_level,
    priority=event_type in (EventType.SUPER_CHAT, EventType.GUARD_BUY),
  )


class DanmakuPushHost:
  """接收上游 POST snapshot 的推送式弹幕入口。"""

  def __init__(
    self,
    studio,
    *,
    host: str = "0.0.0.0",
    port: int = 9100,
    path: str = "/snapshot",
    dedupe_maxlen: int = 2000,
  ) -> None:
    self._studio = studio
    self._host = host
    self._port = port
    self._path = self._normalize_path(path)
    self._dedupe_maxlen = max(100, dedupe_maxlen)
    self._runner: Optional[web.AppRunner] = None
    self._seen: set[tuple[Any, ...]] = set()
    self._seen_order: deque[tuple[Any, ...]] = deque()
    self._request_count = 0
    self._accepted_count = 0
    self._duplicate_count = 0
    self._bad_request_count = 0
    self._empty_request_count = 0

  @staticmethod
  def _normalize_path(path: str) -> str:
    normalized = str(path or "/snapshot").strip() or "/snapshot"
    if not normalized.startswith("/"):
      normalized = "/" + normalized
    return normalized

  async def start(self) -> None:
    if self._runner is not None:
      return
    app = web.Application()
    routes = {self._path, "/"}
    for route in routes:
      app.router.add_post(route, self._handle_push)
    self._runner = web.AppRunner(app)
    await self._runner.setup()
    site = web.TCPSite(self._runner, self._host, self._port)
    await site.start()
    print(f"[弹幕Push] 接收端启动: http://{self._host}:{self._port}{self._path}")
    if self._path != "/":
      print(f"[弹幕Push] 兼容根路径: http://{self._host}:{self._port}/")

  async def stop(self) -> None:
    if self._runner is None:
      return
    await self._runner.cleanup()
    self._runner = None

  def _remember_fingerprint(self, fingerprint: tuple[Any, ...]) -> None:
    if fingerprint in self._seen:
      return
    self._seen.add(fingerprint)
    self._seen_order.append(fingerprint)
    while len(self._seen_order) > self._dedupe_maxlen:
      stale = self._seen_order.popleft()
      self._seen.discard(stale)

  async def _handle_push(self, request: web.Request) -> web.Response:
    self._request_count += 1
    request_index = self._request_count
    try:
      payload = await request.json()
    except Exception as e:
      self._bad_request_count += 1
      raw_body = (await request.text())[:200]
      logger.warning("弹幕 Push JSON 解析失败: %s | body=%s", e, raw_body)
      print(f"[弹幕Push] req#{request_index} 无效 JSON: {e}")
      return web.json_response(
        {"ok": False, "error": "invalid_json", "request_index": request_index},
        status=400,
      )

    items, server_time = parse_snapshot_payload(payload)
    raw_counts = {
      "danmakus": len(payload.get("danmakus", []) or []) if isinstance(payload, dict) else 0,
      "notifications": len(payload.get("notifications", []) or []) if isinstance(payload, dict) else 0,
      "gifts": len(payload.get("gifts", []) or []) if isinstance(payload, dict) else 0,
      "super_chats": len(payload.get("super_chats", []) or []) if isinstance(payload, dict) else 0,
      "guard_buys": len(payload.get("guard_buys", []) or []) if isinstance(payload, dict) else 0,
    }
    parsed_counts = Counter(item.event_type for item in items)
    accepted_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()

    for item in items:
      fingerprint = item_fingerprint(item)
      if fingerprint in self._seen:
        duplicate_counts[item.event_type] += 1
        self._duplicate_count += 1
        continue
      self._remember_fingerprint(fingerprint)
      comment = remote_item_to_comment(item)
      self._studio.send_comment(comment)
      accepted_counts[item.event_type] += 1
      self._accepted_count += 1

    raw_total = sum(raw_counts.values())
    accepted_total = sum(accepted_counts.values())
    duplicate_total = sum(duplicate_counts.values())
    if raw_total == 0 and accepted_total == 0 and duplicate_total == 0:
      self._empty_request_count += 1
    else:
      print(
        "[弹幕Push] req#%d raw=%s parsed=%s accepted=%s dup=%s"
        % (
          request_index,
          raw_counts,
          dict(parsed_counts),
          dict(accepted_counts),
          dict(duplicate_counts),
        )
      )

    ack = {
      "ok": True,
      "request_index": request_index,
      "server_time": server_time,
      "received": len(items),
      "accepted": sum(accepted_counts.values()),
      "duplicates": sum(duplicate_counts.values()),
      "raw_counts": raw_counts,
      "parsed_counts": dict(parsed_counts),
      "accepted_counts": dict(accepted_counts),
      "duplicate_counts": dict(duplicate_counts),
      "totals": {
        "requests": self._request_count,
        "accepted": self._accepted_count,
        "duplicates": self._duplicate_count,
        "bad_requests": self._bad_request_count,
        "empty_requests": self._empty_request_count,
      },
    }
    return web.json_response(ack)

  def debug_state(self) -> dict[str, Any]:
    return {
      "host": self._host,
      "port": self._port,
      "path": self._path,
      "totals": {
        "requests": self._request_count,
        "accepted": self._accepted_count,
        "duplicates": self._duplicate_count,
        "bad_requests": self._bad_request_count,
        "empty_requests": self._empty_request_count,
      },
      "dedupe_cache_size": len(self._seen_order),
    }
