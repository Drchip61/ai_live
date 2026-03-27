"""
游戏解说接入 Host

接收外部游戏解说文本，入队 SpeechQueue，完播后向上游发送 done 回调。

数据流:
  POST /game {"text": "..."} → SpeechQueue(priority=GAME) → TTS 播放 → POST done_url
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import requests
from aiohttp import web

logger = logging.getLogger(__name__)


class GameCommentaryHost:
  """
  游戏解说 HTTP 接入端点。

  其他服务 POST {"text": "一句话"} 到 /game，
  本端将其入队 SpeechQueue，TTS 完播后向 done_url 发送完成信号。
  """

  def __init__(
    self,
    studio,
    *,
    host: str = "0.0.0.0",
    port: int = 6666,
    done_url: str = "http://10.81.7.165:6667",
    done_timeout: float = 5.0,
  ):
    self._studio = studio
    self._host = host
    self._port = port
    self._done_url = done_url
    self._done_timeout = done_timeout
    self._runner: Optional[web.AppRunner] = None
    self._background_tasks: set[asyncio.Task] = set()

  async def start(self) -> None:
    app = web.Application()
    app.router.add_post("/game", self._handle_game)
    self._runner = web.AppRunner(app)
    await self._runner.setup()
    site = web.TCPSite(self._runner, self._host, self._port)
    await site.start()
    print(f"[游戏解说] 监听 http://{self._host}:{self._port}/game")
    print(f"[游戏解说] 完播回调 → {self._done_url}")

  async def stop(self) -> None:
    for task in self._background_tasks:
      task.cancel()
    if self._background_tasks:
      await asyncio.wait(self._background_tasks, timeout=5.0)
    self._background_tasks.clear()
    if self._runner:
      await self._runner.cleanup()
      self._runner = None

  async def _handle_game(self, request: web.Request) -> web.Response:
    try:
      data = await request.json()
    except Exception:
      return web.Response(status=400, text="invalid json")

    text = str(data.get("commentary", "")).strip()
    if not text:
      return web.Response(status=400, text="empty text")

    from streaming_studio.speech_queue import PRIORITY_GAME

    def _on_played(item):
      task = asyncio.create_task(self._send_done(text))
      self._background_tasks.add(task)
      task.add_done_callback(self._background_tasks.discard)

    response_id = await self._studio.enqueue_external_speech(
      text=text,
      source="game",
      priority=PRIORITY_GAME,
      on_played=_on_played,
    )
    if response_id is None:
      return web.Response(status=503, text="speech queue not ready")

    import json as _json
    body = _json.dumps(
      {"status": "queued", "response_id": response_id},
      ensure_ascii=False,
    )
    return web.Response(text=body, content_type="application/json")

  async def _send_done(self, text: str = "") -> None:
    preview = text[:30] if text else ""
    try:
      loop = asyncio.get_running_loop()
      resp = await loop.run_in_executor(
        None,
        lambda: requests.post(
          self._done_url,
          json={"status": "done"},
          timeout=self._done_timeout,
        ),
      )
      print(f"[游戏解说] done → {self._done_url} [{resp.status_code}] «{preview}»")
    except Exception as e:
      print(f"[游戏解说] done 发送失败 → {self._done_url}: {e}")
      logger.error("游戏解说 done 回调失败: %s", e)
