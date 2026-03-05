"""
远程数据源模式入口
从上游服务器拉取截图和弹幕，经 VLM 管线处理后广播到 TTS 服务

用法:
  python run_remote.py --speech-url http://10.81.7.115:9200/say

示例:
  python run_remote.py \
    --screenshot-url http://10.81.7.114:8000/screenshot \
    --danmaku-url http://10.81.7.114:8000/danmaku \
    --persona karin \
    --model anthropic \
    --speech-url http://10.81.7.115:9200/say \
    --frame-interval 5.0
"""

import argparse
import asyncio
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio
from streaming_studio.models import EventType
from connection import SpeechBroadcaster, RemoteSource


def parse_args():
  parser = argparse.ArgumentParser(
    description="远程数据源模式 — 拉取上游截图+弹幕 → VLM 管线 → TTS 广播",
  )

  parser.add_argument(
    "--screenshot-url",
    default="http://10.81.7.114:8000/screenshot",
    help="截图接口 URL（默认 http://10.81.7.114:8000/screenshot）",
  )
  parser.add_argument(
    "--danmaku-url",
    default=None,
    help="弹幕接口 URL（如 http://10.81.7.114:8000/danmaku），不指定则只看画面",
  )

  parser.add_argument(
    "--persona", default="karin",
    choices=["karin", "sage", "kuro", "naixiong", "dacongming"],
    help="主播人设（默认 karin）",
  )
  parser.add_argument(
    "--model", default="anthropic",
    choices=["openai", "anthropic", "gemini"],
    help="模型提供者（默认 anthropic）",
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
    help="弹幕拉取间隔/秒（默认 1.0）",
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
    "--global-memory", action="store_true", default=False,
    help="开启全局记忆（持久化到文件，默认关闭）",
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

  return parser.parse_args()


async def main():
  args = parse_args()

  model_map = {
    "openai": ModelType.OPENAI,
    "anthropic": ModelType.ANTHROPIC,
    "gemini": ModelType.GEMINI,
  }
  model_type = model_map[args.model]

  print("=" * 60)
  print("  AI Live — 远程数据源模式")
  print("=" * 60)
  print(f"  截图: {args.screenshot_url}")
  print(f"  弹幕: {args.danmaku_url or '未配置'}")
  print(f"  人设: {args.persona}")
  print(f"  模型: {args.model} ({args.model_name or '默认'})")
  print(f"  帧间隔: {args.frame_interval}s")
  print(f"  弹幕间隔: {args.danmaku_interval}s")
  print(f"  记忆: {'启用' if not args.no_memory else '禁用'}")
  print(f"  全局记忆: {'启用' if args.global_memory else '禁用'}")
  print(f"  语音服务: {args.speech_url or '未启用'}")
  print(f"  完播同步: {f'port={args.callback_port}' if args.callback_port else '未启用'}")
  print("=" * 60)
  print()

  remote = RemoteSource(
    screenshot_url=args.screenshot_url,
    danmaku_url=args.danmaku_url,
    frame_interval=args.frame_interval,
    danmaku_interval=args.danmaku_interval,
    max_width=args.max_width,
  )

  enable_memory = not args.no_memory
  studio = StreamingStudio(
    persona=args.persona,
    model_type=model_type,
    model_name=args.model_name,
    enable_memory=enable_memory,
    enable_global_memory=args.global_memory,
    video_player=remote,
  )
  studio.enable_streaming = True

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
    )
    speech.attach(studio)

  try:
    if speech:
      await speech.start()
    await studio.start()
    print("[直播开始] 远程数据源运行中... 按 Ctrl+C 停止\n")

    input_task = asyncio.create_task(_input_loop(studio))

    while studio.is_running:
      await asyncio.sleep(1)

  except KeyboardInterrupt:
    print("\n\n[手动停止]")
  finally:
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
