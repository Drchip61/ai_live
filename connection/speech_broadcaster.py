"""
语音服务广播器
将主播回复拆分后发送给语音合成+动作控制服务

主对话模型默认只输出中文，本模块在发往 TTS 前按规则决定：
- `text` 始终使用中文播报
- `text_ja` 仅作为日语字幕字段，按需调用本地 Qwen 生成

支持完播同步：配置 callback_port 后，本端启动 HTTP 服务器接收 TTS 完播回调，
主循环在语音播放完毕前不会生成下一条回复。

数据流:
  StreamerResponse
    → 按 #[motion][emotion][voice_emotion] 标签拆分为片段
    → 从每段提取 text_zh（中文）和 text_ja（已有日语或待补译）
    → 统一固定为中文播报
    → 对需要字幕的片段调用本地翻译，补齐 text_ja
    → 逐段 POST 到语音 API（segment + batch_id + callback_url）
    → 等待 TTS 完播回调（可选）
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import re
import socket
import time
import uuid
from typing import Optional

import requests
from aiohttp import web
from langchain_core.messages import HumanMessage

from langchain_wrapper import ModelType, ModelProvider
from streaming_studio.models import StreamerResponse

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"#\[([^\]]*)\]\[([^\]]*)\](?:\[([^\]]*)\])?")
_EXPRESSION_TAG_RE = re.compile(r"#\[[^\]]*\]\[[^\]]*\](?:\[[^\]]*\])?")
_DEFAULT_EMOTION = "neutral"
_DEFAULT_MOTION = "idle"
_DEFAULT_VOICE_EMOTION = "neutral"

_CHINESE_SPEECH_RE = re.compile(
  r"谢谢|感谢|多谢|谢啦|太感[谢动]|欢迎|来[了啦]"
  r"|舰长|提督|总督|上舰|SC|礼物|打赏"
)

_CLAUSE_SPLIT_RE = re.compile(r"[，,、；;]")
_MOOD_PARTICLE_RE = re.compile(r"^[啊呀哦噢嗯呐呢吧哈嘿嘻唉哎呜嘛欸哇喂哼嘁呵嚯噫诶嗨咦嗷噗嘶呃]+$")
_MAX_CLAUSE_LEN = 8

_CONTINUATION_MOTION_MAP: dict[str, str] = {
  "wave": "nod",
  "dance": "nod",
  "clap": "nod",
  "hands_up": "nod",
  "fists_up": "nod",
  "peace_sign": "nod",
  "leg_raise": "nod",
  "acting_cute": "nod",
  "half_squat": "nod",
  "point_camera": "nod",
  "pointing": "nod",

  "thinking": "chin_rest",
  "chin_pinch": "chin_rest",
  "hand_on_chin": "chin_rest",
  "finger_on_chin": "chin_rest",
  "hands_on_chin": "chin_rest",
  "cheek_rest": "chin_rest",

  "hands_cover_face": "glance_down",
  "face_rest": "glance_down",
  "glance_down": "idle",
  "hands_behind_back": "idle",
  "shush": "idle",

  "arms_crossed": "idle",
  "hands_on_hips": "idle",
  "shrugging": "idle",
  "stop": "idle",
  "hands_raise": "idle",

  "head_shake": "idle",
  "head_tilt": "nod",
  "nod": "idle",
  "look_around": "idle",
  "look_left_panic": "idle",
  "look_right": "idle",
  "eye_roll": "idle",
  "eye_rub": "idle",
  "praying": "idle",
  "stretch": "idle",
  "cold": "idle",
  "disdain": "idle",
  "affirm": "nod",
  "arms_open": "idle",
}
_DEFAULT_CONTINUATION_MOTION = "idle"

_TRANSLATE_JA_SINGLE = (
  "/no_think\n"
  "你是中日翻译器。将以下中文翻译成自然的日语口语。"
  "保持说话者的语气和情感。只输出日语译文，不要任何解释、说明或思考过程。\n\n{text}"
)

_TRANSLATE_JA_BATCH = (
  "/no_think\n"
  "你是中日翻译器。将以下中文句子逐行翻译成自然的日语口语。"
  "保持说话者的语气和情感。保持原始编号和行数，每行只输出日语译文，不要解释或思考过程。\n\n{lines}"
)

_TRANSLATE_EN_SINGLE = (
  "/no_think\n"
  "You are a Chinese to English translator. Translate the following Chinese into natural spoken English. "
  "Only output the translation, with no explanation or reasoning.\n\n{text}"
)

_TRANSLATE_EN_BATCH = (
  "/no_think\n"
  "You are a Chinese to English translator. Translate the following Chinese lines into natural spoken English. "
  "Preserve the original numbering and line count. Only output the translated lines, with no explanation or reasoning.\n\n{lines}"
)


def _detect_local_ip() -> str:
  """探测本机局域网 IP（用于构建 callback_url）"""
  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip
  except Exception:
    return "127.0.0.1"


class SpeechBroadcaster:
  """
  监听主播回复，在发往 TTS 前补齐最终播报文本。

  将 StreamerResponse 按 #[motion][emotion][voice_emotion] 标签拆分为片段，
  固定使用中文播报，并为需要字幕的片段补齐日语 text_ja，再逐段 POST 到语音 API。

  配置 callback_port 后启用完播同步：
    - 本端启动 HTTP 服务器接收 TTS 完播回调
    - 主循环在语音播放完毕前不会生成下一条回复

  发送格式:
    POST {api_url}
    {
      "batch_id": "uuid",
      "callback_url": "http://host:port/speech_done",
      "timestamp": 1234567890.0,
      "segments": [
        {"text": "中文", "text_zh": "中文", "text_ja": "日語",
         "emotion": "脸红", "motion": "Idle", "voice_emotion": "joy"},
        ...
      ]
    }

  完播回调:
    POST {callback_url}
    {"batch_id": "uuid", "status": "done"}
  """

  def __init__(
    self,
    api_url: str,
    model_type: ModelType = ModelType.OPENAI,
    timeout: float = 10.0,
    enabled: bool = True,
    callback_port: Optional[int] = None,
    callback_host: Optional[str] = None,
    translator_enabled: bool = True,
    translator_model_name: str = "Qwen/Qwen3-8B",
    translator_base_url: Optional[str] = None,
  ):
    self._url = api_url
    self._timeout = timeout
    self._background_tasks: set[asyncio.Task] = set()
    self.enabled = enabled
    self._translator_enabled = translator_enabled
    self._translator_model_name = translator_model_name
    self._translator_base_url = translator_base_url
    self._translator = None
    if translator_enabled:
      translator_kwargs = {}
      if translator_base_url:
        translator_kwargs["base_url"] = translator_base_url
      self._translator = ModelProvider().get_model(
        ModelType.LOCAL_QWEN,
        model_name=translator_model_name,
        **translator_kwargs,
      )

    # 完播回调
    self._callback_port = callback_port
    self._callback_host = callback_host
    self._callback_url: Optional[str] = None
    self._playback_done = asyncio.Event()
    self._playback_done.set()
    self._pending_batch_id: Optional[str] = None
    self._total_chars_sent: int = 0
    self._estimated_playback_seconds: float = 0.0
    self._runner: Optional[web.AppRunner] = None
    self._http_executor = ThreadPoolExecutor(
      max_workers=1,
      thread_name_prefix="tts_http",
    )
    self._http_session = requests.Session()
    self._http_backlog: int = 0
    self._send_count: int = 0
    self._wait_interrupted_count: int = 0
    self._playback_interrupted: bool = False

    # 最新回复（供外部轮询）
    self._latest_response: Optional[dict] = None
    self._prepared_response_cache: dict[str, dict] = {}

    # SpeechQueue 模式：旧 _on_response 回调跳过 TTS 发送（由 Dispatcher 接管）
    self._queue_mode = False

  @property
  def url(self) -> str:
    return self._url

  # ── 生命周期 ──────────────────────────────────────────

  async def start(self) -> None:
    """启动完播回调 HTTP 服务器（需要 callback_port）"""
    if not self._callback_port:
      return

    if not self._callback_host:
      self._callback_host = _detect_local_ip()
    self._callback_url = (
      f"http://{self._callback_host}:{self._callback_port}/speech_done"
    )

    app = web.Application()
    app.router.add_post("/speech_done", self._handle_speech_done)
    app.router.add_get("/latest_response", self._handle_latest_response)
    self._runner = web.AppRunner(app)
    await self._runner.setup()
    site = web.TCPSite(self._runner, "0.0.0.0", self._callback_port)
    await site.start()

    print(f"[语音广播] 回调服务器启动: 0.0.0.0:{self._callback_port}")
    print(f"[语音广播] 回调 URL: {self._callback_url}")

    await self._self_test_callback()

  async def _self_test_callback(self) -> None:
    """启动自检：自己 POST 到回调 URL，验证 HTTP 端口可达。"""
    import aiohttp
    test_payload = {"batch_id": "__self_test__", "status": "self_test"}
    try:
      async with aiohttp.ClientSession() as session:
        async with session.post(
          self._callback_url, json=test_payload, timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
          if resp.status == 200:
            print(f"[语音广播] 回调自检通过 ✓ (本机 → {self._callback_url})")
          else:
            print(f"[语音广播] 回调自检异常: HTTP {resp.status}")
    except Exception as e:
      print(f"[语音广播] 回调自检失败 ✗ 本机无法访问 {self._callback_url}: {e}")

  async def stop(self) -> None:
    """停止回调服务器并取消后台任务"""
    for task in self._background_tasks:
      task.cancel()
    if self._background_tasks:
      await asyncio.wait(self._background_tasks, timeout=5.0)
    self._background_tasks.clear()

    if self._runner:
      await self._runner.cleanup()
      self._runner = None

    self._playback_done.set()

    if self._http_session:
      self._http_session.close()

  # ── 挂载与同步 ──────────────────────────────────────

  def attach(self, studio) -> None:
    """注册到 StreamingStudio 的回复回调，并注入语音完播门控 + broadcaster 引用"""
    studio.on_response(self._on_response)
    studio.on_response(self._update_latest_response)
    studio.set_speech_gate(self.wait_for_playback)
    studio.set_speech_broadcaster(self)
    state = "开启" if self.enabled else "关闭（待手动开启）"
    translator_state = (
      f"{self._translator_model_name} @ "
      f"{self._translator_base_url or 'LOCAL_QWEN_BASE_URL / http://localhost:8000/v1'}"
      if self._translator_enabled else "关闭"
    )
    print(f"[语音广播] 已挂载，目标: {self._url}，状态: {state}")
    print(f"[语音广播] 本地日语字幕补译: {translator_state}")
    if self._callback_port:
      print(f"[语音广播] 完播同步: 已启用 (port={self._callback_port})")
    else:
      print(f"[语音广播] 完播同步: 未启用（fire-and-forget 模式）")

  async def wait_for_playback(self) -> str:
    """等待当前语音播放完毕（由 StreamingStudio 主循环调用）。
    超时根据 TTS 队列总字符数动态计算：chars/3 + 15s，下限 30s。
    lookahead 期间可能追加字符，因此用轮询方式动态延伸截止时间。"""
    if self._playback_done.is_set():
      return "interrupted" if self._playback_interrupted else "completed"
    start = time.monotonic()
    deadline = max(30.0, self._total_chars_sent / 3.0 + 15.0)
    while not self._playback_done.is_set():
      deadline = max(deadline, self._total_chars_sent / 3.0 + 15.0)
      elapsed = time.monotonic() - start
      remaining = deadline - elapsed
      if remaining <= 0:
        print(
          f"[语音广播] {deadline:.0f}s 未收到完播回调，疑似回调丢失，强制继续 "
          f"(已发 {self._total_chars_sent} 字)"
        )
        self._playback_interrupted = False
        self._playback_done.set()
        break
      try:
        await asyncio.wait_for(
          self._playback_done.wait(), timeout=min(remaining, 5.0),
        )
      except asyncio.TimeoutError:
        continue
    if self._playback_interrupted:
      self._wait_interrupted_count += 1
      return "interrupted"
    return "completed"

  # ── 回调处理 ──────────────────────────────────────────

  def _on_response(self, response: StreamerResponse) -> None:
    """同步回调入口，关闭时或 queue 模式跳过（Dispatcher 接管 TTS 发送）"""
    if not self.enabled or self._queue_mode:
      return
    self._pending_batch_id = str(uuid.uuid4())
    self._playback_done.clear()
    try:
      loop = asyncio.get_event_loop()
      if loop.is_running():
        task = asyncio.create_task(self._process(response))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    except RuntimeError:
      self._playback_done.set()

  def cancel_current_playback(self) -> None:
    """强制解除 wait_for_playback 阻塞（高优先级弹幕抢占低优先级播放时使用）。"""
    self._playback_interrupted = True
    self._pending_batch_id = f"interrupted:{uuid.uuid4()}"
    self._playback_done.set()

  async def _handle_speech_done(self, request: web.Request) -> web.Response:
    """处理 TTS 完播回调。严格匹配 batch_id，忽略已被抢占的过期回调。"""
    try:
      data = await request.json()
    except Exception as e:
      raw_body = await request.text()
      print(f"[语音广播] 收到无效回调 JSON，仍放行: {e}\n  原始body: {raw_body[:200]}")
      self._playback_done.set()
      return web.Response(status=400, text="invalid json")

    batch_id = data.get("batch_id")
    status = data.get("status", "done")

    if batch_id == "__self_test__":
      return web.Response(text="ok")

    if batch_id and self._pending_batch_id and batch_id != self._pending_batch_id:
      print(
        f"[语音广播] 忽略过期回调 batch={batch_id[:8]}… "
        f"(当前等待 {self._pending_batch_id[:8]}…)"
      )
      return web.Response(text="ok")

    if batch_id and batch_id == self._pending_batch_id:
      print(f"[语音广播] 收到完播回调 batch={batch_id[:8]}… status={status}")
    else:
      print(f"[语音广播] 收到就绪信号 status={status}")

    self._playback_interrupted = False
    self._playback_done.set()
    return web.Response(text="ok")

  def _update_latest_response(self, response: StreamerResponse) -> None:
    """回复回调：更新最新回复供 HTTP 端点返回"""
    prepared = self._prepared_response_cache.get(response.id, {})
    spoken_text_zh = str(
      prepared.get("spoken_text_zh")
      or self._extract_chinese(response.content)
      or ""
    ).strip()
    subtitle_text_ja = str(prepared.get("subtitle_text_ja") or "").strip()
    subtitle_complete = bool(prepared.get("subtitle_complete"))
    self._latest_response = {
      "id": response.id,
      "text": spoken_text_zh,
      "spoken_text_zh": spoken_text_zh,
      "text_ja": subtitle_text_ja,
      "subtitle_text_ja": subtitle_text_ja,
      "subtitle_complete": subtitle_complete,
      "raw": response.content,
      "timestamp": response.timestamp.isoformat(),
      "response_style": response.response_style,
      "reply_target_text": response.reply_target_text,
      "nickname": response.nickname,
    }
    if response.controller_trace:
      self._latest_response["controller_source"] = response.controller_trace.get("source", "")
      self._latest_response["controller_latency_ms"] = response.controller_trace.get("latency_ms", 0.0)
      self._latest_response["controller_plan_json"] = response.controller_trace.get("plan_json")
    if response.timing_trace:
      self._latest_response["timing_trace"] = response.timing_trace

  def _remember_prepared_response(
    self,
    response: StreamerResponse,
    segments: list[dict],
  ) -> None:
    spoken_text_zh = "".join(
      str(seg.get("text_zh") or "").strip()
      for seg in segments
      if str(seg.get("text_zh") or "").strip()
    )
    subtitle_parts: list[str] = []
    subtitle_complete = True
    for seg in segments:
      text_zh = str(seg.get("text_zh") or "").strip()
      text_ja = str(seg.get("text_ja") or "").strip()
      if text_zh and not text_ja:
        subtitle_complete = False
      if text_ja:
        subtitle_parts.append(text_ja)
    self._prepared_response_cache[response.id] = {
      "spoken_text_zh": spoken_text_zh or self._extract_chinese(response.content),
      "subtitle_text_ja": "".join(subtitle_parts),
      "subtitle_complete": subtitle_complete and bool(subtitle_parts),
    }
    if len(self._prepared_response_cache) > 100:
      stale_ids = list(self._prepared_response_cache)[:-50]
      for rid in stale_ids:
        self._prepared_response_cache.pop(rid, None)

  async def prepare_segments_for_broadcast(
    self,
    response: StreamerResponse,
    segments: list[dict],
  ) -> list[dict]:
    prepared = await self._prepare_segments_for_tts(
      [dict(seg) for seg in segments]
    )
    for seg in prepared:
      seg["reply_target_text"] = response.reply_target_text
      seg["nickname"] = response.nickname
    self._remember_prepared_response(response, prepared)
    return prepared

  async def _handle_latest_response(self, request: web.Request) -> web.Response:
    """GET /latest_response — 返回最新回复供弹幕机器人轮询"""
    import json as _json
    if self._latest_response is None:
      body = _json.dumps({"id": None}, ensure_ascii=False)
    else:
      body = _json.dumps(self._latest_response, ensure_ascii=False)
    return web.Response(text=body, content_type="application/json")

  # ── SpeechQueue 模式：单段 / 批量发送 ───────────────────

  async def send_segment(self, segment: dict) -> bool:
    """发送单个 TTS 段。是 send_segments 的单条快捷方式。"""
    return await self.send_segments([segment])

  async def send_segments(self, segments: list[dict]) -> bool:
    """
    批量发送同一 batch 的多个 TTS 段（由 TTS Dispatcher 调用）。

    所有 segment 共享同一 batch_id，连续 POST 不等完播，
    只在最后一段标记 is_last=True。TTS 队列清空后回调该 batch_id。
    调用方需在之后 await wait_for_playback() 等待完播。

    Returns:
      True 表示全部 POST 成功（2xx），False 表示任一失败
    """
    if not self.enabled or not segments:
      return False

    self._send_count += 1
    self._pending_batch_id = str(uuid.uuid4())
    self._playback_interrupted = False
    self._playback_done.clear()
    self._total_chars_sent = 0

    prepared = await self._prepare_segments_for_tts(
      [dict(s) for s in segments]
    )

    total = len(prepared)
    all_ok = True
    for i, seg in enumerate(prepared):
      self._total_chars_sent += len(str(seg.get("text_zh", "") or ""))
      seg["timestamp"] = time.time()
      seg["batch_id"] = self._pending_batch_id
      seg["seq"] = i
      seg["total"] = total
      seg["is_last"] = (i == total - 1)
      if self._callback_url:
        seg["callback_url"] = self._callback_url
      ok = await self._run_http_call(
        self._post_segment,
        self._strip_private_segment_fields(seg),
      )
      if not ok:
        all_ok = False

    if not all_ok or not self._callback_url:
      self._playback_interrupted = False
      self._playback_done.set()
    return all_ok

  async def _send_lookahead(self, segment: dict) -> bool:
    """
    在当前 batch 播放期间预发后续 segment。

    更新 _pending_batch_id 为新 batch_id。TTS 队列按序播放所有段，
    队列清空后发送最后一个 batch_id 的完播回调，刚好匹配。
    """
    if not self.enabled:
      return False

    self._pending_batch_id = str(uuid.uuid4())

    seg = (await self._prepare_segments_for_tts([dict(segment)]))[0]
    self._total_chars_sent += len(str(seg.get("text_zh", "") or ""))
    seg["timestamp"] = time.time()
    seg["batch_id"] = self._pending_batch_id
    seg["seq"] = 0
    seg["total"] = 1
    seg["is_last"] = True
    if self._callback_url:
      seg["callback_url"] = self._callback_url

    return await self._run_http_call(
      self._post_segment,
      self._strip_private_segment_fields(seg),
    )

  # ── 处理管线（旧路径，_on_response 回调使用）──────────

  async def _process(self, response: StreamerResponse) -> None:
    """解析标签并补齐最终播报文本后逐条发送。"""
    try:
      t_start = time.monotonic()
      segments = self._parse_segments(response.content)
      if not segments:
        self._playback_done.set()
        return

      self._apply_chinese_speech(segments, response.response_style)
      segments = self._split_long_segments(segments)
      segments = await self.prepare_segments_for_broadcast(response, segments)
      self._update_latest_response(response)

      total = len(segments)
      all_ok = True
      for i, seg in enumerate(segments):
        seg["timestamp"] = time.time()
        seg["batch_id"] = self._pending_batch_id
        seg["seq"] = i
        seg["total"] = total
        seg["is_last"] = (i == total - 1)
        if self._callback_url:
          seg["callback_url"] = self._callback_url
        ok = await self._run_http_call(
          self._post_segment,
          self._strip_private_segment_fields(seg),
        )
        if not ok:
          all_ok = False

      t_done = time.monotonic()
      total_ms = (t_done - t_start) * 1000
      print(f"[语音广播耗时] 发送:{total_ms:.0f}ms ({total}段)")

      total_chars = sum(len(s.get("text", "")) for s in segments)
      self._estimated_playback_seconds = total_chars / 3.5 + 2.0

      if not self._callback_url or not all_ok:
        self._playback_interrupted = False
        self._playback_done.set()
    except Exception as e:
      logger.error("SpeechBroadcaster 处理失败: %s", e)
      print(f"[语音广播] 处理失败: {e}")
      self._playback_done.set()

  # ── 中文语音替换 ─────────────────────────────────────

  @staticmethod
  def _apply_chinese_speech(
    segments: list[dict], response_style: str,
  ) -> None:
    """当前 TTS 永远使用中文播报，text_ja 仅作为字幕字段。"""
    _ = response_style
    for seg in segments:
      text_zh = (seg.get("text_zh") or seg.get("text") or "").strip()
      seg["text_zh"] = text_zh
      seg["text"] = text_zh
      seg["language"] = "Chinese"

  async def _prepare_segments_for_tts(self, segments: list[dict]) -> list[dict]:
    """固定中文播报；必要时补齐 text_ja 字幕字段。"""
    if not segments:
      return segments
    if not self._translator_enabled:
      for seg in segments:
        text_zh = (seg.get("text_zh") or seg.get("text") or "").strip()
        seg["text_zh"] = text_zh
        seg["text"] = text_zh
        seg["text_ja"] = ""
        seg["language"] = "Chinese"
        seg["_subtitle_state"] = "missing"
      return segments
    if all(seg.get("_subtitle_state") in ("ready", "missing") for seg in segments):
      for seg in segments:
        text_zh = (seg.get("text_zh") or seg.get("text") or "").strip()
        seg["text_zh"] = text_zh
        seg["text"] = text_zh
        seg["language"] = "Chinese"
      return segments

    translate_indices: list[int] = []
    translate_texts: list[str] = []

    for idx, seg in enumerate(segments):
      text_zh = (seg.get("text_zh") or seg.get("text") or "").strip()
      text_ja = (seg.get("text_ja") or "").strip()
      seg["text_zh"] = text_zh
      seg["text"] = text_zh
      seg["language"] = "Chinese"

      if text_zh and text_ja and text_ja != text_zh:
        seg["_subtitle_state"] = "ready"
        continue

      if text_zh:
        translate_indices.append(idx)
        translate_texts.append(text_zh)
      else:
        seg["text_ja"] = ""
        seg["_subtitle_state"] = "missing"

    if not translate_texts:
      for seg in segments:
        if seg.get("_subtitle_state") not in ("ready", "missing"):
          seg["text_ja"] = str(seg.get("text_ja") or "").strip()
          seg["_subtitle_state"] = "ready" if seg["text_ja"] else "missing"
      return segments

    translated, translation_ok = await self._translate_batch(translate_texts, lang="ja")
    for idx, ja in zip(translate_indices, translated):
      seg = segments[idx]
      text_zh = (seg.get("text_zh") or "").strip()
      text_ja = (ja or "").strip()
      if translation_ok and text_ja:
        seg["text_ja"] = text_ja
        seg["_subtitle_state"] = "ready"
      else:
        seg["text_ja"] = ""
        seg["_subtitle_state"] = "missing"

    return segments

  @staticmethod
  def _extract_chinese(content: str) -> str:
    """剥离表情标签和日语翻译，只保留纯中文文本"""
    parts = _EXPRESSION_TAG_RE.split(content)
    chinese_parts = []
    for part in parts:
      part = part.strip()
      if not part:
        continue
      sep_idx = part.find(" / ")
      if sep_idx >= 0:
        part = part[:sep_idx].strip()
      if part:
        chinese_parts.append(part)
    return "".join(chinese_parts) if chinese_parts else content

  # ── 解析 / 翻译 / 发送 ──────────────────────────────

  @staticmethod
  def _split_bilingual(text: str) -> tuple[str, str]:
    """从 '中文 / 日語' 格式提取双语，无 ' / ' 时降级为中文"""
    sep = " / "
    idx = text.find(sep)
    if idx >= 0:
      zh = text[:idx].strip()
      ja = text[idx + len(sep):].strip()
      return (zh, ja) if (zh and ja) else (text.strip(), text.strip())
    return (text.strip(), text.strip())

  @staticmethod
  def _build_segment(
    raw: str,
    motion: str = _DEFAULT_MOTION,
    emotion: str = _DEFAULT_EMOTION,
    voice_emotion: str = _DEFAULT_VOICE_EMOTION,
  ) -> dict:
    zh, ja = SpeechBroadcaster._split_bilingual(raw)
    return {
      "text": zh,
      "text_zh": zh,
      "text_ja": ja if ja and ja != zh else "",
      "language": "Chinese",
      "emotion": emotion,
      "motion": motion,
      "voice_emotion": voice_emotion,
    }

  @staticmethod
  def _parse_segments(content: str) -> list[dict]:
    """
    按 #[motion][emotion][voice_emotion] 标签拆分文本为片段，并提取内嵌双语。

    格式: "#[Jump][星星][joy]好厉害！ / すごい！#[Nod][- -][serenity]嗯嗯"
      → [
          {"text": "好厉害！", "text_zh": "好厉害！", "text_ja": "すごい！", "emotion": "星星", "motion": "Jump", "voice_emotion": "joy"},
          {"text": "嗯嗯", "text_zh": "嗯嗯", "text_ja": "", "emotion": "- -", "motion": "Nod", "voice_emotion": "serenity"},
        ]

    兼容旧双标签输入：缺第三标签时自动补默认 voice_emotion。
    """
    segments = []
    matches = list(_TAG_RE.finditer(content))

    if not matches:
      clean = _TAG_RE.sub("", content).strip()
      if clean:
        segments.append(SpeechBroadcaster._build_segment(clean))
      return segments

    leading = content[:matches[0].start()].strip()
    if leading:
      segments.append(SpeechBroadcaster._build_segment(leading))

    for idx, match in enumerate(matches):
      motion = match.group(1).strip() or _DEFAULT_MOTION
      emotion = match.group(2).strip() or _DEFAULT_EMOTION
      voice_emotion = (match.group(3) or "").strip() or _DEFAULT_VOICE_EMOTION
      body_start = match.end()
      body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
      raw = content[body_start:body_end].strip()
      if raw:
        segments.append(SpeechBroadcaster._build_segment(
          raw,
          motion=motion,
          emotion=emotion,
          voice_emotion=voice_emotion,
        ))

    return segments

  @staticmethod
  def _split_long_segments(
    segments: list[dict],
    max_clause_len: int = _MAX_CLAUSE_LEN,
  ) -> list[dict]:
    """
    在每个 segment 内部按中文逗号等标点拆分子句。

    规则：
    - 子句超过 max_clause_len 字时独立为新 segment
    - 纯语气词（啊、呢、吧…）不独立，合并到前一个子句
    - 首段保留原始 motion，后续段使用延续动作避免重复触发
    - emotion 和 voice_emotion 全部继承原 segment
    """
    result: list[dict] = []
    for seg in segments:
      text = (seg.get("text_zh") or seg.get("text") or "").strip()
      clauses = _CLAUSE_SPLIT_RE.split(text)

      if len(clauses) <= 1 or len(text) <= max_clause_len:
        result.append(seg)
        continue

      merged: list[str] = []
      for clause in clauses:
        clause = clause.strip()
        if not clause:
          continue
        if _MOOD_PARTICLE_RE.match(clause) and merged:
          merged[-1] += "，" + clause
        else:
          merged.append(clause)

      # 句首语气词向后合并（"啊，好厉害" → "啊，好厉害"）
      if len(merged) > 1 and _MOOD_PARTICLE_RE.match(merged[0]):
        merged[1] = merged[0] + "，" + merged[1]
        merged.pop(0)

      if len(merged) <= 1:
        result.append(seg)
        continue

      buf = ""
      sub_segments: list[str] = []
      for clause in merged:
        if not buf:
          buf = clause
        elif len(buf) + len(clause) + 1 <= max_clause_len:
          buf += "，" + clause
        elif _MOOD_PARTICLE_RE.match(buf):
          buf += "，" + clause
        else:
          sub_segments.append(buf)
          buf = clause
      if buf:
        sub_segments.append(buf)

      if len(sub_segments) <= 1:
        result.append(seg)
        continue

      original_motion = seg.get("motion", _DEFAULT_MOTION)
      continuation = _CONTINUATION_MOTION_MAP.get(
        original_motion.lower(), _DEFAULT_CONTINUATION_MOTION,
      )
      for i, sub_text in enumerate(sub_segments):
        new_seg = dict(seg)
        new_seg["text"] = sub_text
        new_seg["text_zh"] = sub_text
        new_seg["text_ja"] = ""
        if i > 0:
          new_seg["motion"] = continuation
        result.append(new_seg)

    return result

  async def _translate_batch(
    self, texts: list[str], lang: str = "ja",
  ) -> tuple[list[str], bool]:
    """用小模型批量翻译，返回 (译文列表, 是否调用成功)。"""
    if not texts:
      return [], True
    if self._translator is None:
      return texts, False

    if lang == "en":
      tpl_single, tpl_batch = _TRANSLATE_EN_SINGLE, _TRANSLATE_EN_BATCH
    else:
      tpl_single, tpl_batch = _TRANSLATE_JA_SINGLE, _TRANSLATE_JA_BATCH

    if len(texts) == 1:
      prompt = tpl_single.format(text=texts[0])
    else:
      numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
      prompt = tpl_batch.format(lines=numbered)

    try:
      result = await self._translator.ainvoke([HumanMessage(content=prompt)])
      raw = result.content if hasattr(result, "content") else str(result)
      translated = self._parse_translation(raw, len(texts))
      logger.info("翻译完成 [%s]: %s → %s", lang, texts, translated)
      return translated, True
    except Exception as e:
      logger.error("翻译失败 [%s]，降级为中文: %s", lang, e)
      print(f"[语音广播] 翻译失败 [{lang}]，降级为中文: {e}")
      return texts, False

  @staticmethod
  def _parse_translation(raw: str, expected: int) -> list[str]:
    """解析翻译结果，提取逐行译文"""
    if expected == 1:
      cleaned = re.sub(r"^(?:\d+[\.\、\)\]\s]+|[-*]\s+)", "", raw.strip()).strip()
      return [cleaned]

    lines = []
    for line in raw.strip().splitlines():
      line = line.strip()
      if not line:
        continue
      cleaned = re.sub(r"^(?:\d+[\.\、\)\]\s]+|[-*]\s+)", "", line).strip()
      if cleaned:
        lines.append(cleaned)

    if len(lines) == expected:
      return lines
    if len(lines) > expected:
      return lines[:expected]
    while len(lines) < expected:
      lines.append("")
    return lines

  @staticmethod
  def _strip_private_segment_fields(segment: dict) -> dict:
    return {
      key: value
      for key, value in segment.items()
      if not str(key).startswith("_")
    }

  def _post_segment(self, segment: dict) -> bool:
    """POST 单条语音段到语音服务，返回是否成功（2xx）"""
    import json as _json

    def _preview(text: str, limit: int = 120) -> str:
      text = str(text or "")
      if len(text) <= limit:
        return text
      return text[:limit] + "...<仅截断日志预览>"

    try:
      logger.info(
        "发送语音段:\n%s",
        _json.dumps(segment, ensure_ascii=False, indent=2),
      )
      resp = self._http_session.post(
        self._url, json=segment, timeout=self._timeout,
      )
      status = resp.status_code
      bid = segment["batch_id"][:8]
      seq = segment["seq"]
      total = segment["total"]
      zh = str(segment.get("text_zh", "") or "")
      ja = str(segment.get("text_ja", "") or "")
      spoken = str(segment.get("text", "") or "")
      language = segment.get("language", "Chinese")
      print(
        f"[语音广播] [{status}] batch={bid}… ({seq+1}/{total})\n"
        f"  [{segment['motion']}][{segment['emotion']}][{segment.get('voice_emotion', _DEFAULT_VOICE_EMOTION)}] "
        f"lang: {language} | payload_chars zh={len(zh)} ja={len(ja)} tts={len(spoken)}\n"
        f"  zh_preview: {_preview(zh) or '-'}\n"
        f"  ja_preview: {_preview(ja) or '-'}\n"
        f"  tts_preview: {_preview(spoken) or '-'}"
      )
      if status >= 400:
        print(f"[语音广播] TTS 返回 {status}，跳过等待回调")
      return status < 400
    except requests.RequestException as e:
      logger.error("[语音广播] POST 失败: %s", e)
      print(f"[语音广播] POST 失败: {e}")
      return False

  async def _run_http_call(self, callback, *args):
    """将阻塞式 HTTP 发送固定在专用单线程 executor，避免抢占默认线程池。"""
    loop = asyncio.get_running_loop()
    self._http_backlog += 1
    try:
      return await loop.run_in_executor(
        self._http_executor,
        partial(callback, *args),
      )
    finally:
      self._http_backlog = max(0, self._http_backlog - 1)
