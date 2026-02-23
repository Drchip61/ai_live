"""
VLM 直播间 GUI
NiceGUI Web 界面，集成视频画面、弹幕侧栏、用户输入和控制台输出

用法:
  python run_vlm_gui.py --video <视频文件> --danmaku <弹幕XML>

示例:
  python run_vlm_gui.py \
    --video data/sample.mp4 \
    --danmaku data/sample.xml \
    --persona kuro \
    --model anthropic
"""

import argparse
import collections
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from nicegui import ui, context

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio, Comment
from streaming_studio.models import ResponseChunk
from video_source import VideoPlayer


def parse_args():
  parser = argparse.ArgumentParser(
    description="VLM 直播间 GUI — 视频画面 + 弹幕 → 多模态理解（NiceGUI 版）",
  )
  parser.add_argument(
    "--video", required=True,
    help="视频文件路径（mp4/mkv/flv 等 OpenCV 支持的格式）",
  )
  parser.add_argument(
    "--danmaku", default=None,
    help="B站弹幕 XML 文件路径（可选）",
  )
  parser.add_argument(
    "--persona", default="karin",
    choices=["karin", "sage", "kuro"],
    help="主播人设（默认 karin）",
  )
  parser.add_argument(
    "--model", default="anthropic",
    choices=["openai", "anthropic"],
    help="模型提供者（默认 anthropic）",
  )
  parser.add_argument(
    "--model-name", default=None,
    help="指定模型名称（默认使用提供者的大模型）",
  )
  parser.add_argument(
    "--speed", type=float, default=1.0,
    help="播放速度倍率（默认 1.0）",
  )
  parser.add_argument(
    "--frame-interval", type=float, default=5.0,
    help="帧采样间隔/秒（默认 5.0）",
  )
  parser.add_argument(
    "--max-width", type=int, default=1280,
    help="帧最大宽度（默认 1280）",
  )
  parser.add_argument(
    "--no-memory", action="store_true", default=False,
    help="禁用分层记忆系统",
  )
  parser.add_argument(
    "--global-memory", action="store_true", default=False,
    help="开启全局记忆",
  )
  parser.add_argument(
    "--topic-manager", action="store_true", default=False,
    help="启用话题管理器",
  )
  parser.add_argument(
    "--port", type=int, default=8081,
    help="Web 服务端口（默认 8081）",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  model_map = {
    "openai": ModelType.OPENAI,
    "anthropic": ModelType.ANTHROPIC,
  }
  model_type = model_map[args.model]

  player = VideoPlayer(
    video_path=args.video,
    danmaku_path=args.danmaku,
    speed=args.speed,
    frame_interval=args.frame_interval,
    max_width=args.max_width,
  )

  studio = StreamingStudio(
    persona=args.persona,
    model_type=model_type,
    model_name=args.model_name,
    enable_memory=not args.no_memory,
    enable_global_memory=args.global_memory,
    enable_topic_manager=args.topic_manager,
    video_player=player,
  )
  studio.enable_streaming = True

  print(f"VLM 直播间 GUI")
  print(f"  视频: {args.video} ({player.duration:.1f}s)")
  print(f"  弹幕: {args.danmaku or '无'}")
  print(f"  人设: {args.persona}  模型: {args.model}")
  print(f"  端口: {args.port}")

  @ui.page("/")
  async def vlm_page():
    # ── 页面局部状态 ──
    streaming_labels: dict[str, ui.label] = {}
    streamed_ids: collections.deque[str] = collections.deque(maxlen=50)
    danmaku_entries: list[ui.element] = []
    callback_refs = {
      "response_fn": None,
      "chunk_fn": None,
      "pre_fn": None,
      "frame_fn": None,
      "danmaku_fn": None,
    }

    DANMAKU_HEIGHT = "height: calc(55vh - 120px)"
    CONSOLE_HEIGHT = "height: calc(35vh - 50px)"

    # ── 顶部控制栏 ──
    with ui.row().classes(
      "w-full items-center gap-4 px-4 py-2 bg-gray-100 shrink-0"
    ):
      ui.label("VLM 直播间").classes("text-xl font-bold")

      async def on_start():
        start_btn.disable()
        _register_callbacks()
        await studio.start()
        stop_btn.enable()

      async def on_stop():
        stop_btn.disable()
        _cleanup_callbacks()
        await studio.stop()
        start_btn.enable()
        progress.set_value(0)

      start_btn = ui.button("开始", on_click=on_start).props(
        "dense icon=play_arrow color=green"
      )
      stop_btn = ui.button("停止", on_click=on_stop).props(
        "dense icon=stop color=red"
      )
      stop_btn.disable()

      progress = ui.linear_progress(value=0, show_value=False).classes("flex-1")
      time_label = ui.label(
        f"0.0 / {player.duration:.1f}s"
      ).classes("text-sm text-gray-600 whitespace-nowrap")

    # ── 中间区域：视频画面 + 弹幕侧栏 ──
    with ui.row().classes("w-full gap-0 px-2").style("height: 55vh"):

      # 左侧：视频画面
      with ui.column().classes("h-full p-1 overflow-hidden").style("flex: 3"):
        video_img = ui.image().classes("w-full h-full object-contain").style(
          "background: #111"
        )

      # 右侧：弹幕侧栏 + 输入
      with ui.column().classes("h-full gap-2 pl-2 pr-1 py-1").style("flex: 2"):
        ui.label("弹幕").classes("text-sm font-bold text-green-700 shrink-0")

        with ui.scroll_area().classes(
          "w-full bg-gray-50 rounded border flex-1"
        ).style(DANMAKU_HEIGHT) as danmaku_scroll:
          danmaku_column = ui.column().classes("w-full gap-1 p-2")

        # 输入栏
        with ui.row().classes("w-full gap-2 shrink-0"):
          msg_input = ui.input(placeholder="输入弹幕...").props(
            "dense outlined"
          ).classes("flex-1")

          def send_comment():
            content = msg_input.value.strip()
            if not content or not studio.is_running:
              return
            comment = Comment(
              user_id="manual_user",
              nickname="手动观众",
              content=content,
            )
            studio.send_comment(comment)
            _add_danmaku("手动观众", content, priority=True)
            msg_input.value = ""

          ui.button("发送", on_click=send_comment).props("dense")
          msg_input.on("keydown.enter", send_comment)

    # ── 底部：控制台 ──
    with ui.column().classes("w-full gap-1 px-2 py-1").style(
      "height: calc(45vh - 60px)"
    ):
      ui.label("控制台").classes("text-sm font-bold text-blue-700 shrink-0")

      with ui.scroll_area().classes(
        "w-full bg-gray-900 rounded flex-1"
      ).style(CONSOLE_HEIGHT) as console_scroll:
        console_column = ui.column().classes("w-full gap-0 p-2")

    # ── 辅助函数 ──

    def _add_danmaku(nickname: str, content: str, priority: bool = False):
      """添加弹幕条目到侧栏"""
      while len(danmaku_entries) >= 200:
        old = danmaku_entries.pop(0)
        old.delete()

      with danmaku_column:
        if priority:
          lbl = ui.label(f"\u2605 {nickname}: {content}").classes(
            "text-sm text-orange-600"
          )
        else:
          lbl = ui.label(f"{nickname}: {content}").classes(
            "text-sm text-gray-700"
          )
        danmaku_entries.append(lbl)
      danmaku_scroll.scroll_to(percent=1.0)

    def _add_console_block(
      text: str,
      classes: str = "text-sm text-gray-300 font-mono whitespace-pre-wrap",
    ):
      """添加一行到控制台"""
      with console_column:
        ui.label(text).classes(classes)
      console_scroll.scroll_to(percent=1.0)

    # ── 回调函数 ──

    def on_frame(frame):
      """视频新帧 → 更新画面"""
      video_img.set_source(f"data:image/jpeg;base64,{frame.base64_jpeg}")

    def on_danmaku(danmaku):
      """视频弹幕到达 → 侧栏显示"""
      nick = (
        f"观众{danmaku.user_hash[:4]}"
        if danmaku.user_hash
        else "观众"
      )
      _add_danmaku(nick, danmaku.content)

    def on_pre_response(old_comments, new_comments):
      """生成回复前 → 控制台显示弹幕列表"""
      lines = ["--- 回复弹幕 ---"]
      for c in old_comments:
        tag = "[优先]" if c.priority else "[旧]"
        lines.append(f"{tag} {c.nickname}: {c.content}")
      for c in new_comments:
        tag = "[优先]" if c.priority else "[新]"
        lines.append(f"{tag} {c.nickname}: {c.content}")
      lines.append("-" * 16)
      _add_console_block(
        "\n".join(lines),
        "text-sm text-yellow-400 font-mono whitespace-pre-wrap",
      )

    def on_chunk(chunk: ResponseChunk):
      """流式回复 → 控制台逐字显示"""
      if chunk.response_id not in streaming_labels:
        with console_column:
          lbl = ui.label(chunk.accumulated).classes(
            "text-sm text-green-400 font-mono whitespace-pre-wrap"
          )
        streaming_labels[chunk.response_id] = lbl
        console_scroll.scroll_to(percent=1.0)
      else:
        lbl = streaming_labels[chunk.response_id]
        lbl.set_text(chunk.accumulated)
        console_scroll.scroll_to(percent=1.0)

      if chunk.done:
        lbl = streaming_labels.pop(chunk.response_id, None)
        if lbl is not None:
          lbl.classes(
            replace="text-sm text-gray-200 font-mono whitespace-pre-wrap"
          )
        streamed_ids.append(chunk.response_id)

    def on_response(response):
      """完整回复回调（非流式兜底）"""
      if response.id in streamed_ids:
        return
      _add_console_block(
        response.content,
        "text-sm text-gray-200 font-mono whitespace-pre-wrap",
      )

    # ── 回调注册 / 清理 ──

    def _register_callbacks():
      callback_refs["frame_fn"] = on_frame
      callback_refs["danmaku_fn"] = on_danmaku
      callback_refs["pre_fn"] = on_pre_response
      callback_refs["chunk_fn"] = on_chunk
      callback_refs["response_fn"] = on_response

      player.on_frame(on_frame)
      player.on_danmaku(on_danmaku)
      studio.on_pre_response(on_pre_response)
      studio.on_response_chunk(on_chunk)
      studio.on_response(on_response)

    def _cleanup_callbacks():
      fn = callback_refs.get("frame_fn")
      if fn and fn in player._on_frame_callbacks:
        player._on_frame_callbacks.remove(fn)
      callback_refs["frame_fn"] = None

      fn = callback_refs.get("danmaku_fn")
      if fn and fn in player._on_danmaku_callbacks:
        player._on_danmaku_callbacks.remove(fn)
      callback_refs["danmaku_fn"] = None

      fn = callback_refs.get("pre_fn")
      if fn and fn in studio._pre_response_callbacks:
        studio._pre_response_callbacks.remove(fn)
      callback_refs["pre_fn"] = None

      fn = callback_refs.get("chunk_fn")
      if fn:
        studio.remove_chunk_callback(fn)
      callback_refs["chunk_fn"] = None

      fn = callback_refs.get("response_fn")
      if fn:
        studio.remove_callback(fn)
      callback_refs["response_fn"] = None

      streaming_labels.clear()

    context.client.on_disconnect(_cleanup_callbacks)

    # ── 进度条定时器 ──

    def update_progress():
      if not studio.is_running:
        return
      current = player.current_sec
      total = player.duration
      if total > 0:
        progress.set_value(current / total)
        time_label.set_text(f"{current:.1f} / {total:.1f}s")
      if player.is_finished:
        time_label.set_text(f"{total:.1f} / {total:.1f}s (完毕)")

    ui.timer(0.5, update_progress)

  ui.run(port=args.port, reload=False, title="VLM 直播间")


if __name__ == "__main__":
  main()
