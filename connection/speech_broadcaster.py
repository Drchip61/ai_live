"""
语音服务广播器
将主播回复翻译成日语，连同表情/动作标签发送给语音合成+动作控制服务

数据流:
  StreamerResponse
    → 按 #[motion][emotion] 标签拆分为片段
    → 批量翻译中文 → 日语（小模型）
    → 逐条 POST 到语音 API
"""

import asyncio
import logging
import re
import time
from typing import Optional

import requests
from langchain_core.messages import HumanMessage

from langchain_wrapper import ModelType, ModelProvider
from streaming_studio.models import StreamerResponse

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"#\[([^\]]*)\]\[([^\]]*)\]")
_DEFAULT_EMOTION = "- -"
_DEFAULT_MOTION = "Idle"

_TRANSLATE_PROMPT_SINGLE = (
  "你是中日翻译器。将以下中文翻译成自然的日语口语。"
  "保持说话者的语气和情感。只输出日语译文，不要任何解释。\n\n{text}"
)

_TRANSLATE_PROMPT_BATCH = (
  "你是中日翻译器。将以下中文句子逐行翻译成自然的日语口语。"
  "保持说话者的语气和情感。保持原始编号和行数，每行只输出日语译文。\n\n{lines}"
)


class SpeechBroadcaster:
  """
  监听主播回复，翻译成日语后发送给语音/动作控制服务。

  将 StreamerResponse 按 #[motion][emotion] 标签拆分为片段，
  批量翻译为日语，逐条 POST 到语音 API。

  API 格式:
    POST {api_url}
    {"text": "日語", "emotion": "脸红", "motion": "Idle", "timestamp": 1234567890.0}
  """

  def __init__(
    self,
    api_url: str,
    model_type: ModelType = ModelType.OPENAI,
    timeout: float = 10.0,
  ):
    self._url = api_url
    self._timeout = timeout
    self._model = ModelProvider.remote_small(model_type)
    self._background_tasks: set[asyncio.Task] = set()

  def attach(self, studio) -> None:
    """注册到 StreamingStudio 的回复回调"""
    studio.on_response(self._on_response)
    print(f"[语音广播] 已挂载，目标: {self._url}")

  def _on_response(self, response: StreamerResponse) -> None:
    """同步回调入口，启动异步处理任务"""
    try:
      loop = asyncio.get_event_loop()
      if loop.is_running():
        task = asyncio.create_task(self._process(response))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
    except RuntimeError:
      pass

  async def _process(self, response: StreamerResponse) -> None:
    """解析 → 翻译 → 逐条发送"""
    try:
      segments = self._parse_segments(response.content)
      if not segments:
        return

      zh_texts = [s["text"] for s in segments]
      ja_texts = await self._translate_batch(zh_texts)

      for seg, ja_text in zip(segments, ja_texts):
        seg["text"] = ja_text
        seg["timestamp"] = time.time()
        await asyncio.to_thread(self._post_segment, seg)
    except Exception as e:
      logger.error("SpeechBroadcaster 处理失败: %s", e)
      print(f"[语音广播] 处理失败: {e}")

  @staticmethod
  def _parse_segments(content: str) -> list[dict]:
    """
    按 #[motion][emotion] 标签拆分文本为片段。

    每个标签应用于其后面的文本；标签前的文本使用默认值。
    例: "#[Jump][星星]好厉害！#[Nod][- -]嗯嗯"
      → [
          {"text": "好厉害！", "emotion": "星星", "motion": "Jump"},
          {"text": "嗯嗯", "emotion": "- -", "motion": "Nod"},
        ]
    """
    parts = _TAG_RE.split(content)
    segments = []

    leading = parts[0].strip()
    if leading:
      segments.append({
        "text": leading,
        "emotion": _DEFAULT_EMOTION,
        "motion": _DEFAULT_MOTION,
      })

    i = 1
    while i + 2 < len(parts):
      motion = parts[i].strip() or _DEFAULT_MOTION
      emotion = parts[i + 1].strip() or _DEFAULT_EMOTION
      text = parts[i + 2].strip()
      if text:
        segments.append({
          "text": text,
          "emotion": emotion,
          "motion": motion,
        })
      i += 3

    if not segments:
      clean = _TAG_RE.sub("", content).strip()
      if clean:
        segments.append({
          "text": clean,
          "emotion": _DEFAULT_EMOTION,
          "motion": _DEFAULT_MOTION,
        })

    return segments

  async def _translate_batch(self, texts: list[str]) -> list[str]:
    """用小模型批量翻译中文 → 日语"""
    if not texts:
      return []

    if len(texts) == 1:
      prompt = _TRANSLATE_PROMPT_SINGLE.format(text=texts[0])
    else:
      numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
      prompt = _TRANSLATE_PROMPT_BATCH.format(lines=numbered)

    try:
      result = await self._model.ainvoke([HumanMessage(content=prompt)])
      raw = result.content if hasattr(result, "content") else str(result)
      ja_texts = self._parse_translation(raw, len(texts))
      logger.info("翻译完成: %s → %s", texts, ja_texts)
      return ja_texts
    except Exception as e:
      logger.error("翻译失败，降级为中文: %s", e)
      print(f"[语音广播] 翻译失败，降级为中文: {e}")
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

  def _post_segment(self, segment: dict) -> None:
    """POST 单个片段到语音服务"""
    try:
      resp = requests.post(
        self._url, json=segment, timeout=self._timeout,
      )
      status = resp.status_code
      print(
        f"[语音广播] [{status}] "
        f"emotion={segment['emotion']} motion={segment['motion']} "
        f"text={segment['text'][:50]}"
      )
    except requests.RequestException as e:
      logger.error("[语音广播] POST 失败: %s", e)
      print(f"[语音广播] POST 失败: {e}")
