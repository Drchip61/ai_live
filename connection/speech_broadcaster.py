"""
语音服务广播器
将主播回复（内嵌中日双语）拆分后发送给语音合成+动作控制服务

主对话模型直接输出 '中文 / 日語' 格式，本模块只做解析和发送，无翻译步骤。

支持完播同步：配置 callback_port 后，本端启动 HTTP 服务器接收 TTS 完播回调，
主循环在语音播放完毕前不会生成下一条回复。

数据流:
  StreamerResponse
    → 按 #[motion][emotion] 标签拆分为片段
    → 从每段提取 text_zh（中文）和 text（日语）
    → 逐段 POST 到语音 API（segment + batch_id + callback_url）
    → 等待 TTS 完播回调（可选）
"""

import asyncio
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

_TAG_RE = re.compile(r"#\[([^\]]*)\]\[([^\]]*)\]")
_DEFAULT_EMOTION = "- -"
_DEFAULT_MOTION = "Idle"

_TRANSLATE_JA_SINGLE = (
  "你是中日翻译器。将以下中文翻译成自然的日语口语。"
  "保持说话者的语气和情感。只输出日语译文，不要任何解释。\n\n{text}"
)

_TRANSLATE_JA_BATCH = (
  "你是中日翻译器。将以下中文句子逐行翻译成自然的日语口语。"
  "保持说话者的语气和情感。保持原始编号和行数，每行只输出日语译文。\n\n{lines}"
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
  监听主播回复，翻译后整体发送给语音/动作控制服务。

  将 StreamerResponse 按 #[motion][emotion] 标签拆分为片段，
  批量翻译为日语+英语，整体 POST 到语音 API。

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
        {"text": "日語", "text_zh": "中文",
         "emotion": "脸红", "motion": "Idle"},
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
  ):
    self._url = api_url
    self._timeout = timeout
    self._model = ModelProvider.remote_small(model_type)
    self._background_tasks: set[asyncio.Task] = set()
    self.enabled = enabled

    # 完播回调
    self._callback_port = callback_port
    self._callback_host = callback_host
    self._callback_url: Optional[str] = None
    self._playback_done = asyncio.Event()
    self._playback_done.set()
    self._pending_batch_id: Optional[str] = None
    self._estimated_playback_seconds: float = 0.0
    self._runner: Optional[web.AppRunner] = None

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
    self._runner = web.AppRunner(app)
    await self._runner.setup()
    site = web.TCPSite(self._runner, "0.0.0.0", self._callback_port)
    await site.start()

    print(f"[语音广播] 回调服务器启动: 0.0.0.0:{self._callback_port}")
    print(f"[语音广播] 回调 URL: {self._callback_url}")

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

  # ── 挂载与同步 ──────────────────────────────────────

  def attach(self, studio) -> None:
    """注册到 StreamingStudio 的回复回调，并注入语音完播门控"""
    studio.on_response(self._on_response)
    studio.set_speech_gate(self.wait_for_playback)
    state = "开启" if self.enabled else "关闭（待手动开启）"
    print(f"[语音广播] 已挂载，目标: {self._url}，状态: {state}")
    if self._callback_port:
      print(f"[语音广播] 完播同步: 已启用 (port={self._callback_port})")
    else:
      print(f"[语音广播] 完播同步: 未启用（fire-and-forget 模式）")

  async def wait_for_playback(self) -> None:
    """等待当前语音播放完毕（由 StreamingStudio 主循环调用）"""
    if self._playback_done.is_set():
      return
    timeout = min(max(self._estimated_playback_seconds * 2, 15.0), 60.0)
    try:
      await asyncio.wait_for(self._playback_done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
      logger.warning("语音播放等待超时 (%ss)，强制继续", timeout)
      print(f"[语音广播] 等待超时 ({timeout:.0f}s)，强制继续")
      self._playback_done.set()

  # ── 回调处理 ──────────────────────────────────────────

  def _on_response(self, response: StreamerResponse) -> None:
    """同步回调入口，关闭时跳过"""
    if not self.enabled:
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

  async def _handle_speech_done(self, request: web.Request) -> web.Response:
    """处理 TTS 完播回调（任何合法请求都解除阻塞）"""
    try:
      data = await request.json()
    except Exception:
      return web.Response(status=400, text="invalid json")

    batch_id = data.get("batch_id")
    status = data.get("status", "done")

    if batch_id and batch_id == self._pending_batch_id:
      print(f"[语音广播] 收到完播回调 batch={batch_id[:8]}… status={status}")
    elif self._pending_batch_id:
      print(f"[语音广播] 收到完播回调 (batch_id 不匹配，仍放行) status={status}")
    else:
      print(f"[语音广播] 收到就绪信号 status={status}")

    self._playback_done.set()
    return web.Response(text="ok")

  # ── 处理管线 ──────────────────────────────────────────

  async def _process(self, response: StreamerResponse) -> None:
    """解析内嵌双语 → 逐条发送（无翻译步骤）"""
    try:
      t_start = time.monotonic()
      segments = self._parse_segments(response.content)
      if not segments:
        self._playback_done.set()
        return

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
        ok = await asyncio.to_thread(self._post_segment, seg)
        if not ok:
          all_ok = False

      t_done = time.monotonic()
      total_ms = (t_done - t_start) * 1000
      print(f"[语音广播耗时] 发送:{total_ms:.0f}ms ({total}段)")

      total_chars = sum(len(s.get("text", "")) for s in segments)
      self._estimated_playback_seconds = total_chars / 3.5 + 2.0

      if not self._callback_url or not all_ok:
        self._playback_done.set()
    except Exception as e:
      logger.error("SpeechBroadcaster 处理失败: %s", e)
      print(f"[语音广播] 处理失败: {e}")
      self._playback_done.set()

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
  def _parse_segments(content: str) -> list[dict]:
    """
    按 #[motion][emotion] 标签拆分文本为片段，并提取内嵌双语。

    格式: "#[Jump][星星]好厉害！ / すごい！#[Nod][- -]嗯嗯"
      → [
          {"text": "すごい！", "text_zh": "好厉害！", "emotion": "星星", "motion": "Jump"},
          {"text": "嗯嗯", "text_zh": "嗯嗯", "emotion": "- -", "motion": "Nod"},
        ]
    """
    parts = _TAG_RE.split(content)
    segments = []

    leading = parts[0].strip()
    if leading:
      zh, ja = SpeechBroadcaster._split_bilingual(leading)
      segments.append({
        "text": ja,
        "text_zh": zh,
        "emotion": _DEFAULT_EMOTION,
        "motion": _DEFAULT_MOTION,
      })

    i = 1
    while i + 2 < len(parts):
      motion = parts[i].strip() or _DEFAULT_MOTION
      emotion = parts[i + 1].strip() or _DEFAULT_EMOTION
      raw = parts[i + 2].strip()
      if raw:
        zh, ja = SpeechBroadcaster._split_bilingual(raw)
        segments.append({
          "text": ja,
          "text_zh": zh,
          "emotion": emotion,
          "motion": motion,
        })
      i += 3

    if not segments:
      clean = _TAG_RE.sub("", content).strip()
      if clean:
        zh, ja = SpeechBroadcaster._split_bilingual(clean)
        segments.append({
          "text": ja,
          "text_zh": zh,
          "emotion": _DEFAULT_EMOTION,
          "motion": _DEFAULT_MOTION,
        })

    return segments

  async def _translate_batch(
    self, texts: list[str], lang: str = "ja",
  ) -> list[str]:
    """用小模型批量翻译，lang: 'ja' 或 'en'"""
    if not texts:
      return []

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
      result = await self._model.ainvoke([HumanMessage(content=prompt)])
      raw = result.content if hasattr(result, "content") else str(result)
      translated = self._parse_translation(raw, len(texts))
      logger.info("翻译完成 [%s]: %s → %s", lang, texts, translated)
      return translated
    except Exception as e:
      logger.error("翻译失败 [%s]，降级为中文: %s", lang, e)
      print(f"[语音广播] 翻译失败 [{lang}]，降级为中文: {e}")
      return texts

  @staticmethod
  def _parse_translation(raw: str, expected: int) -> list[str]:
    """解析翻译结果，提取逐行译文"""
    if expected == 1:
      return [raw.strip()]

    lines = []
    for line in raw.strip().splitlines():
      line = line.strip()
      if not line:
        continue
      cleaned = re.sub(r"^\d+[\.\、\)\]\s]+", "", line).strip()
      if cleaned:
        lines.append(cleaned)

    if len(lines) == expected:
      return lines
    if len(lines) > expected:
      return lines[:expected]
    while len(lines) < expected:
      lines.append("")
    return lines

  def _post_segment(self, segment: dict) -> bool:
    """POST 单条语音段到语音服务，返回是否成功（2xx）"""
    import json as _json
    try:
      logger.info(
        "发送语音段:\n%s",
        _json.dumps(segment, ensure_ascii=False, indent=2),
      )
      resp = requests.post(
        self._url, json=segment, timeout=self._timeout,
      )
      status = resp.status_code
      bid = segment["batch_id"][:8]
      seq = segment["seq"]
      total = segment["total"]
      zh = segment.get("text_zh", "")[:60]
      ja = segment["text"][:60]
      print(
        f"[语音广播] [{status}] batch={bid}… ({seq+1}/{total})\n"
        f"  [{segment['emotion']}][{segment['motion']}] "
        f"zh: {zh} / ja: {ja}"
      )
      if status >= 400:
        print(f"[语音广播] TTS 返回 {status}，跳过等待回调")
      return status < 400
    except requests.RequestException as e:
      logger.error("[语音广播] POST 失败: %s", e)
      print(f"[语音广播] POST 失败: {e}")
      return False
