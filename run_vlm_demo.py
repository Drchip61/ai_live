"""
VLM 直播间 Demo
加载 B站视频 + 弹幕 XML，用 Claude VLM 做视觉+弹幕理解的虚拟主播

用法:
  python run_vlm_demo.py --video <视频文件> --danmaku <弹幕XML>

示例:
  python run_vlm_demo.py \
    --video data/sample.mp4 \
    --danmaku data/sample.xml \
    --persona karin \
    --speed 2.0

B站弹幕获取方式:
  1. 找到视频的 cid（在视频页面 F12 → Network 搜索 "cid"）
  2. 下载弹幕 XML: https://comment.bilibili.com/{cid}.xml
  3. 或使用 bilibili-dl / you-get 等工具同时下载视频和弹幕
"""

import argparse
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio
from video_source import VideoPlayer


def parse_args():
  parser = argparse.ArgumentParser(
    description="VLM 直播间 Demo — 视频画面 + 弹幕 → Claude 多模态理解",
  )
  parser.add_argument(
    "--video", required=True,
    help="视频文件路径（mp4/mkv/flv 等 OpenCV 支持的格式）",
  )
  parser.add_argument(
    "--danmaku", default=None,
    help="B站弹幕 XML 文件路径（可选，不提供则只看画面不读弹幕）",
  )
  parser.add_argument(
    "--persona", default="karin",
    choices=["karin", "sage", "kuro", "naixiong"],
    help="主播人设（默认 karin）",
  )
  parser.add_argument(
    "--model", default="anthropic",
    choices=["openai", "anthropic", "gemini"],
    help="模型提供者（VLM 推荐 anthropic，默认 anthropic）",
  )
  parser.add_argument(
    "--model-name", default=None,
    help="指定模型名称（默认使用提供者的大模型）",
  )
  parser.add_argument(
    "--speed", type=float, default=1.0,
    help="播放速度倍率（默认 1.0 实时，2.0 = 两倍速）",
  )
  parser.add_argument(
    "--frame-interval", type=float, default=5.0,
    help="帧采样间隔/秒（默认 5.0，越小画面更新越频繁但 token 消耗更大）",
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
    "--topic-manager", action="store_true", default=False,
    help="启用话题管理器",
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
  print("  VLM 虚拟直播间 Demo")
  print("=" * 60)
  print(f"  视频: {args.video}")
  print(f"  弹幕: {args.danmaku or '无'}")
  print(f"  人设: {args.persona}")
  print(f"  模型: {args.model} ({args.model_name or '默认'})")
  print(f"  速度: {args.speed}x")
  print(f"  帧间隔: {args.frame_interval}s")
  print(f"  记忆: {'启用' if not args.no_memory else '禁用'}")
  print(f"  全局记忆: {'启用' if args.global_memory else '禁用'}")
  print("=" * 60)
  print()

  player = VideoPlayer(
    video_path=args.video,
    danmaku_path=args.danmaku,
    speed=args.speed,
    frame_interval=args.frame_interval,
    max_width=args.max_width,
  )

  print(f"视频信息: {player.duration:.1f}s, {player._extractor.resolution}")
  print()

  enable_memory = not args.no_memory
  studio = StreamingStudio(
    persona=args.persona,
    model_type=model_type,
    model_name=args.model_name,
    enable_memory=enable_memory,
    enable_global_memory=args.global_memory,
    enable_topic_manager=args.topic_manager,
    video_player=player,
  )
  studio.enable_streaming = True

  def on_chunk(chunk):
    if chunk.done:
      print()
      print()
    else:
      print(chunk.chunk, end="", flush=True)

  def on_response(response):
    pass

  def on_pre_response(old_comments, new_comments):
    print("--- 回复弹幕 ---")
    for c in old_comments:
      label = "[优先]" if c.priority else "[旧]"
      print(f"{label} {c.nickname}: {c.content}")
    for c in new_comments:
      label = "[优先]" if c.priority else "[新]"
      print(f"{label} {c.nickname}: {c.content}")
    print("-" * 16)

  studio.on_pre_response(on_pre_response)
  studio.on_response_chunk(on_chunk)
  studio.on_response(on_response)

  try:
    await studio.start()
    print("[直播开始] 视频播放中... 按 Ctrl+C 停止\n")

    # 同时接受终端手动弹幕输入
    input_task = asyncio.create_task(_input_loop(studio))

    while studio.is_running:
      await asyncio.sleep(1)
      if player.is_finished:
        print("\n[视频播放完毕]")
        await asyncio.sleep(5)
        break

  except KeyboardInterrupt:
    print("\n\n[手动停止]")
  finally:
    await studio.stop()
    print("直播间已关闭")


async def _input_loop(studio: StreamingStudio):
  """允许用户在终端手动发送弹幕（模拟额外观众）"""
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
