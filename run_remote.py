"""
远程数据源模式入口
从上游服务器拉取截图，并在本地接收推送式弹幕，经 VLM 管线处理后广播到 TTS 服务

用法:
  python run_remote.py --speech-url http://10.81.7.115:9200/say

示例:
  python run_remote.py \
    --screenshot-url http://10.81.7.114:8000/screenshot \
    --persona karin \
    --model openai \
    --speech-url http://10.81.7.115:9200/say \
    --frame-interval 5.0
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType, ModelProvider
from langchain_wrapper.model_provider import REMOTE_MODELS
from llm_controller import LLMController
from streaming_studio import StreamingStudio
from streaming_studio.models import EventType
from connection import DanmakuPushHost, SpeechBroadcaster, RemoteSource, GameCommentaryHost


# 远程模式 Controller：3 路 expert 全走本地 Ollama，零网络延迟。
_LOCAL_CTRL_MODEL = "hoangquan456/qwen3-nothink:8b"
_CONTROLLER_TRANSPORT_TIMEOUT = 3.5


def _controller_model_kwargs() -> dict:
  return {
    "temperature": 0.2,
    "max_tokens": 256,
    "timeout": _CONTROLLER_TRANSPORT_TIMEOUT,
    "max_retries": 0,
  }

def _controller_expert_specs(model_provider: ModelProvider):
  return (
    ("reply_judge", "ReplyJudge+ActionGuard", ModelType.LOCAL_QWEN, _LOCAL_CTRL_MODEL, {}),
    ("context_advisor", "ContextAdvisor", ModelType.LOCAL_QWEN, _LOCAL_CTRL_MODEL, {}),
    ("style_advisor", "StyleAdvisor", ModelType.LOCAL_QWEN, _LOCAL_CTRL_MODEL, {}),
  )


def _build_controller_expert_models(model_provider: ModelProvider):
  expert_models = {}
  expert_labels = {}
  for key, label, model_type, model_name, extra_kwargs in _controller_expert_specs(model_provider):
    expert_models[key] = model_provider.get_model(
      model_type,
      model_name=model_name,
      **_controller_model_kwargs(),
      **extra_kwargs,
    )
    if extra_kwargs.get("base_url"):
      expert_labels[key] = (
        f"{label}: {model_name} via {model_type.value} @ {extra_kwargs['base_url']}"
      )
    else:
      expert_labels[key] = f"{label}: {model_name} via {model_type.value}"
  # ActionGuard 默认并入 ReplyJudge，不再单独发起一次请求。
  expert_models["action_guard"] = None
  return expert_models["reply_judge"], "per-expert", expert_models, expert_labels


def _chat_model_signature(model) -> tuple[str, str, str]:
  """尽量按底层模型地址 + 名称去重，避免重复预热同一实例。"""
  model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or ""
  base_url = (
    getattr(model, "openai_api_base", None)
    or getattr(model, "base_url", None)
    or ""
  )
  if not model_name and not base_url:
    return (type(model).__name__, "", str(id(model)))
  return (type(model).__name__, str(model_name), str(base_url))


async def _warmup_chat_model(label: str, model) -> None:
  """启动时预热本地模型，减少首轮请求的冷启动抖动。"""
  runner = model.bind(max_tokens=1, temperature=0) if hasattr(model, "bind") else model
  try:
    await asyncio.wait_for(
      runner.ainvoke("系统预热请求，只回复 ok。"),
      timeout=45.0,
    )
    print(f"[Warmup] {label} 已就绪")
  except Exception as exc:
    print(f"[Warmup] {label} 预热失败: {exc}")


async def _warmup_controller_models(expert_models: Optional[dict]) -> None:
  if not expert_models:
    return

  tasks = []
  seen: set[tuple[str, str, str]] = set()
  for key, model in expert_models.items():
    if model is None:
      continue
    signature = _chat_model_signature(model)
    if signature in seen:
      continue
    seen.add(signature)
    tasks.append(_warmup_chat_model(key, model))

  if not tasks:
    return

  print("[Warmup] 正在预热本地 Controller 模型...")
  await asyncio.gather(*tasks)


def parse_args():
  parser = argparse.ArgumentParser(
    description="远程数据源模式 — 拉取上游截图 + 接收推送弹幕 → VLM 管线 → TTS 广播",
  )

  parser.add_argument(
    "--screenshot-url",
    default="http://10.81.7.114:8000/screenshot",
    help="截图接口 URL（默认 http://10.81.7.114:8000/screenshot）",
  )
  parser.add_argument(
    "--danmaku-url",
    default=None,
    help="旧轮询弹幕接口 URL（已废弃，仅保留兼容提示；远程模式现改为本地 push 接收）",
  )
  parser.add_argument(
    "--danmaku-host",
    default="0.0.0.0",
    help="推送式弹幕监听 host（默认 0.0.0.0）",
  )
  parser.add_argument(
    "--danmaku-port",
    type=int,
    default=9100,
    help="推送式弹幕监听端口（默认 9100）",
  )
  parser.add_argument(
    "--danmaku-path",
    default="/snapshot",
    help="推送式弹幕接收路径（默认 /snapshot，同时兼容根路径 /）",
  )

  parser.add_argument(
    "--persona", default="karin",
    choices=["karin", "sage", "kuro", "naixiong", "dacongming", "mio"],
    help="主播人设（默认 karin）",
  )
  parser.add_argument(
    "--model", default="openai",
    choices=["openai", "anthropic", "gemini", "deepseek"],
    help="模型提供者（默认 openai）",
  )
  parser.add_argument(
    "--model-name", default=None,
    help="指定模型名称（默认使用提供者的大模型）",
  )

  parser.add_argument(
    "--frame-interval", type=float, default=5.0,
    help="截图拉取间隔/秒（默认 5.0）",
  )
  parser.add_argument(
    "--danmaku-interval", type=float, default=1.0,
    help="旧轮询弹幕拉取间隔/秒（已废弃，仅保留兼容）",
  )
  parser.add_argument(
    "--max-width", type=int, default=1280,
    help="帧最大宽度（默认 1280，降低可节省 token）",
  )

  parser.add_argument(
    "--no-memory", action="store_true", default=False,
    help="禁用分层记忆系统（默认启用）",
  )
  parser.add_argument(
    "--ephemeral-memory", action="store_true", default=False,
    help="使用临时记忆（进程退出后丢弃，默认持久化到文件）",
  )
  parser.add_argument(
    "--state-card", action="store_true", default=False,
    help="启用主播状态卡系统（维护精力/耐心/情绪惯性等状态，默认关闭）",
  )

  parser.add_argument(
    "--speech-url", default=None,
    help="语音/动作服务 URL（如 http://10.81.7.115:9200/say）",
  )

  parser.add_argument(
    "--callback-port", type=int, default=None,
    help="完播回调监听端口（如 9201），启用后 TTS 播完会通知本端，主循环等待播完再生成下一条",
  )
  parser.add_argument(
    "--callback-host", default=None,
    help="回调 URL 中的 host（TTS 用此地址访问本端），默认自动探测局域网 IP",
  )
  parser.add_argument(
    "--disable-local-translation", action="store_true", default=False,
    help="兼容旧参数：禁用本地日语字幕补译（当前远程模式默认已关闭）",
  )
  parser.add_argument(
    "--enable-local-translation", action="store_true", default=False,
    help="显式启用本地日语字幕补译（默认关闭，text_ja 留空）",
  )
  parser.add_argument(
    "--translator-model-name", default="Qwen/Qwen3-8B",
    help="本地翻译模型名称（默认 Qwen/Qwen3-8B）",
  )
  parser.add_argument(
    "--translator-base-url", default=None,
    help="本地日语字幕补译模型 OpenAI 兼容接口地址（默认读取 LOCAL_QWEN_BASE_URL 或 http://localhost:8000/v1）",
  )

  parser.add_argument(
    "--enable-controller", action="store_true", default=False,
    help="启用 Controller（默认走 per-expert wiring，而不是单共享模型）",
  )
  parser.add_argument(
    "--controller-provider", default="openai",
    choices=["openai", "anthropic", "gemini", "deepseek"],
    help="共享 Controller 模型的 provider（仅在显式传入 --controller-model 时生效）",
  )
  parser.add_argument(
    "--controller-url", default=None,
    help="兼容旧参数：任意非空值都只作为启用 Controller 的开关，不再真的连接该 URL",
  )
  parser.add_argument(
    "--controller-model", default=None,
    help="显式指定时使用单共享 Controller 模型；不传则默认启用 per-expert wiring",
  )

  parser.add_argument(
    "--game-port", type=int, default=None,
    help="游戏解说接收端口（如 6666），启用后监听 POST /game 接收外部解说文本",
  )
  parser.add_argument(
    "--game-done-url", default="http://10.81.7.165:6667",
    help="游戏解说完播后发送 done 信号的目标 URL（默认 http://10.81.7.165:6667）",
  )

  return parser.parse_args()


async def main():
  args = parse_args()

  model_map = {
    "openai": ModelType.OPENAI,
    "anthropic": ModelType.ANTHROPIC,
    "gemini": ModelType.GEMINI,
    "deepseek": ModelType.DEEPSEEK,
  }
  model_type = model_map[args.model]
  controller_model_type = model_map[args.controller_provider]
  translator_enabled = bool(args.enable_local_translation) and not args.disable_local_translation
  controller_enabled = bool(args.enable_controller or args.controller_url)
  controller_shared_mode = controller_enabled and bool(args.controller_model)
  controller_model_name = args.controller_model or REMOTE_MODELS[controller_model_type]["small"]
  model_provider = ModelProvider()

  # 主模型 DeepSeek 时使用独立 key，避免与 controller 共用限速
  main_model_kwargs = {}
  if model_type == ModelType.DEEPSEEK:
    ds_key2 = model_provider._get_secret("deepseek_api_key2")
    if ds_key2:
      main_model_kwargs["api_key"] = ds_key2

  # 不支持 VLM 的主模型自动配一个支持图片的备用模型
  _NO_VLM_PROVIDERS = {ModelType.DEEPSEEK}
  vlm_model_type = None
  vlm_model_name = None
  if model_type in _NO_VLM_PROVIDERS:
    vlm_model_type = ModelType.ANTHROPIC
    vlm_model_name = None  # 走默认 Sonnet

  controller_chat_model = None
  expert_models = None
  controller_expert_labels = {}
  if controller_enabled and not controller_shared_mode:
    (
      controller_chat_model,
      controller_model_name,
      expert_models,
      controller_expert_labels,
    ) = _build_controller_expert_models(model_provider)

  print("=" * 60)
  print("  AI Live — 远程数据源模式")
  print("=" * 60)
  print(f"  截图: {args.screenshot_url}")
  print(f"  弹幕接入: push http://{args.danmaku_host}:{args.danmaku_port}{args.danmaku_path}")
  if args.danmaku_url:
    print(f"  旧弹幕URL: {args.danmaku_url}（已废弃，本次不会轮询）")
  print(f"  人设: {args.persona}")
  print(f"  模型: {args.model} ({args.model_name or '默认'})")
  if vlm_model_type:
    _vlm_name = vlm_model_name or REMOTE_MODELS[vlm_model_type]["large"]
    print(f"  VLM 备用: {_vlm_name} via {vlm_model_type.value}（有画面时自动切换）")
  print(f"  帧间隔: {args.frame_interval}s")
  print(f"  记忆: {'启用' if not args.no_memory else '禁用'}")
  print(f"  记忆持久化: {'关闭（临时模式）' if args.ephemeral_memory else '启用'}")
  print(f"  状态卡: {'启用' if args.state_card else '关闭'}")
  print(f"  语音服务: {args.speech_url or '未启用'}")
  print(f"  完播同步: {f'port={args.callback_port}' if args.callback_port else '未启用'}")
  if controller_enabled:
    if controller_shared_mode:
      provider_name = controller_model_type.value
      print(f"  Controller: 启用（共享模型 {controller_model_name} via {provider_name}）")
    else:
      print(f"  Controller: 启用（{controller_model_name} wiring）")
      for label in controller_expert_labels.values():
        print(f"    - {label}")
    if args.controller_url:
      print(f"  Controller旧URL: {args.controller_url}（已忽略，仅作为启用开关）")
  else:
    print("  Controller: 关闭（未启用 --enable-controller，当前使用 fallback 决策）")
  if translator_enabled:
    base_url = args.translator_base_url or "LOCAL_QWEN_BASE_URL / http://localhost:8000/v1"
    print(f"  本地日语字幕补译: {args.translator_model_name} @ {base_url}")
  else:
    print("  本地日语字幕补译: 关闭（text_ja 留空）")
  if args.game_port:
    print(f"  游戏解说: port={args.game_port}  done→{args.game_done_url}")
  print("=" * 60)
  print()

  remote = RemoteSource(
    screenshot_url=args.screenshot_url,
    danmaku_url=None,
    frame_interval=args.frame_interval,
    danmaku_interval=args.danmaku_interval,
    max_width=args.max_width,
  )

  enable_memory = not args.no_memory
  controller = None
  if controller_enabled:
    if controller_shared_mode:
      controller_chat_model = model_provider.get_model(
        controller_model_type,
        model_name=controller_model_name,
        **_controller_model_kwargs(),
      )
    controller = LLMController(
      model=controller_chat_model,
      model_name=controller_model_name,
      expert_models=expert_models,
    )
  studio = StreamingStudio(
    persona=args.persona,
    model_type=model_type,
    model_name=args.model_name,
    model_kwargs=main_model_kwargs or None,
    vlm_model_type=vlm_model_type,
    vlm_model_name=vlm_model_name,
    enable_memory=enable_memory,
    enable_global_memory=not args.ephemeral_memory,
    enable_state_card=args.state_card,
    enable_controller=False,
    controller=controller,
    video_player=remote,
  )
  studio.enable_streaming = True
  danmaku_push = DanmakuPushHost(
    studio,
    host=args.danmaku_host,
    port=args.danmaku_port,
    path=args.danmaku_path,
  )

  def on_chunk(chunk):
    if chunk.done:
      print(f"\n[主播] {chunk.accumulated}\n")

  def on_pre_response(old_comments, new_comments):
    print("--- 回复弹幕 ---")
    for c in old_comments:
      label = "[优先]" if c.priority else "[旧]"
      text = c.format_for_llm() if c.event_type != EventType.DANMAKU else c.content
      print(f"{label} {c.nickname}: {text}")
    for c in new_comments:
      label = "[优先]" if c.priority else "[新]"
      text = c.format_for_llm() if c.event_type != EventType.DANMAKU else c.content
      print(f"{label} {c.nickname}: {text}")
    print("-" * 16)

  studio.on_pre_response(on_pre_response)
  studio.on_response_chunk(on_chunk)

  speech = None
  if args.speech_url:
    speech = SpeechBroadcaster(
      api_url=args.speech_url,
      model_type=model_type,
      callback_port=args.callback_port,
      callback_host=args.callback_host,
      translator_enabled=translator_enabled,
      translator_model_name=args.translator_model_name,
      translator_base_url=args.translator_base_url if translator_enabled else None,
    )
    speech.attach(studio)

  game_host = None
  if args.game_port:
    game_host = GameCommentaryHost(
      studio,
      port=args.game_port,
      done_url=args.game_done_url,
    )

  try:
    if speech:
      await speech.start()
    if controller_enabled and not controller_shared_mode:
      await _warmup_controller_models(expert_models)
    await studio.start()
    await danmaku_push.start()
    if game_host:
      await game_host.start()
    print("[直播开始] 远程数据源运行中... 按 Ctrl+C 停止\n")

    input_task = asyncio.create_task(_input_loop(studio))

    while studio.is_running:
      await asyncio.sleep(1)

  except KeyboardInterrupt:
    print("\n\n[手动停止]")
  finally:
    if game_host:
      await game_host.stop()
    await danmaku_push.stop()
    await studio.stop()
    if speech:
      await speech.stop()
    print("直播间已关闭")


async def _input_loop(studio: StreamingStudio):
  """终端手动弹幕输入（调试用）"""
  from streaming_studio.models import Comment

  loop = asyncio.get_event_loop()
  while studio.is_running:
    try:
      line = await loop.run_in_executor(None, sys.stdin.readline)
      line = line.strip()
      if not line:
        continue
      if line == "/quit":
        break
      comment = Comment(
        user_id="manual_user",
        nickname="手动观众",
        content=line,
      )
      studio.send_comment(comment)
      print(f"  [已发送] {line}")
    except (EOFError, KeyboardInterrupt):
      break
    except Exception:
      break


if __name__ == "__main__":
  asyncio.run(main())
