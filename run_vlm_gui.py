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
import asyncio
import base64
import collections
import sys
from pathlib import Path
from typing import Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from nicegui import ui, context, app
from starlette.responses import StreamingResponse

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio, Comment
from streaming_studio.models import ResponseChunk
from video_source import VideoPlayer
from connection import SpeechBroadcaster


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
    choices=["karin", "sage", "kuro", "naixiong"],
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
  parser.add_argument(
    "--speech-url", default=None,
    help="语音/动作服务 URL（如 http://10.81.7.115:9200/say）",
  )
  return parser.parse_args()


def main():
  args = parse_args()

  model_map = {
    "openai": ModelType.OPENAI,
    "anthropic": ModelType.ANTHROPIC,
    "gemini": ModelType.GEMINI,
  }
  model_type = model_map[args.model]

  # MJPEG 视频流：显示帧通过 HTTP 流式推送，避免 WebSocket base64 闪烁
  _mjpeg_state = {"jpeg_bytes": b"", "running": False}

  def _store_display_frame(frame):
    raw = frame.base64_jpeg
    if isinstance(raw, str) and raw.startswith("data:") and "," in raw:
      raw = raw.split(",", 1)[1]
    try:
      _mjpeg_state["jpeg_bytes"] = base64.b64decode(raw)
    except Exception as e:
      # 容错：异常时保持上一帧，避免画面断流
      print(f"[MJPEG] 解码显示帧失败: {e}", flush=True)

  def _normalize_path(raw_path: str) -> str:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
      p = (project_root / p).resolve()
    return str(p)

  def _build_runtime(video_path: str, danmaku_path: Optional[str]):
    normalized_video = _normalize_path(video_path)
    normalized_danmaku = _normalize_path(danmaku_path) if danmaku_path else None

    player = VideoPlayer(
      video_path=normalized_video,
      danmaku_path=normalized_danmaku,
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
    # 高倍速播放时按比例缩短回复等待时间，维持“每分钟视频内容”的回复密度。
    # 例如 4x 时默认 3~10 秒会缩短到约 0.75~2.5 秒（真实时间）。
    speed_scale = max(1.0, float(args.speed))
    studio.min_interval = max(0.8, studio.min_interval / speed_scale)
    studio.max_interval = max(studio.min_interval + 0.2, studio.max_interval / speed_scale)
    player.on_display_frame(_store_display_frame)

    _terminal_state = {"active_response_id": None}

    def _terminal_chunk_cb(chunk):
      """Always-on terminal callback: buffer streaming, print mapped text on done"""
      active_id = _terminal_state["active_response_id"]
      if chunk.response_id != active_id:
        _terminal_state["active_response_id"] = chunk.response_id
      if chunk.done:
        print(f"[主播] {chunk.accumulated}")
        _terminal_state["active_response_id"] = None

    studio.on_response_chunk(_terminal_chunk_cb)

    if args.speech_url:
      speech = SpeechBroadcaster(api_url=args.speech_url, model_type=model_type)
      speech.attach(studio)

    from collections import deque
    # 队列适当放大，避免高倍速/短时断连时挤掉 done 事件导致“绿色不变白”
    gui_chunk_queue = deque(maxlen=5000)
    gui_pre_response_queue = deque(maxlen=500)
    gui_response_queue = deque(maxlen=500)
    gui_danmaku_queue = deque(maxlen=1000)

    def _gui_chunk_cb(chunk):
      gui_chunk_queue.append(chunk)

    def _gui_pre_response_cb(old_comments, new_comments):
      gui_pre_response_queue.append((old_comments, new_comments))

    def _gui_response_cb(resp):
      gui_response_queue.append(resp)

    studio.on_response_chunk(_gui_chunk_cb)
    studio.on_pre_response(_gui_pre_response_cb)
    studio.on_response(_gui_response_cb)

    def _gui_danmaku_cb(comment):
      gui_danmaku_queue.append(comment)

    player.on_danmaku(_gui_danmaku_cb)

    return {
      "video_path": normalized_video,
      "danmaku_path": normalized_danmaku,
      "player": player,
      "studio": studio,
      "gui_chunk_queue": gui_chunk_queue,
      "gui_pre_response_queue": gui_pre_response_queue,
      "gui_response_queue": gui_response_queue,
      "gui_danmaku_queue": gui_danmaku_queue,
    }

  runtime = _build_runtime(args.video, args.danmaku)

  _placeholder_jpeg = base64.b64decode(
    b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBAQEA8QDw8QDw8PDw8PDw8QDw8PFREWFhURFRUYHSggGBolHRUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OFQ8PFS0dHR0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAQMBIgACEQEDEQH/xAAXAAADAQAAAAAAAAAAAAAAAAAAAQID/8QAFhEBAQEAAAAAAAAAAAAAAAAAAQAC/9oADAMBAAIQAxAAAAG0AP/EABYQAQEBAAAAAAAAAAAAAAAAAAABEf/aAAgBAQABBQJv/8QAFhEBAQEAAAAAAAAAAAAAAAAAABEh/9oACAEDAQE/AVf/xAAVEQEBAAAAAAAAAAAAAAAAAAABEP/aAAgBAgEBPwFX/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJf/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyFf/9k="
  )

  def _get_placeholder_jpeg():
    return _placeholder_jpeg

  async def _mjpeg_generator():
    while True:
      if _mjpeg_state["running"]:
        interval = 1.0 / max(runtime["player"].display_fps, 1)
        data = _mjpeg_state["jpeg_bytes"]
        if data:
          header = (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            + f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
          )
          yield (
            header + data + b"\r\n"
          )
          await asyncio.sleep(interval)
          continue
      placeholder = _get_placeholder_jpeg()
      header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(placeholder)}\r\n\r\n".encode("ascii")
      )
      yield (
        header + placeholder + b"\r\n"
      )
      await asyncio.sleep(1.0)

  @app.get("/video-stream")
  async def video_stream_route():
    return StreamingResponse(
      _mjpeg_generator(),
      media_type="multipart/x-mixed-replace; boundary=frame",
      headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
      },
    )

  print(f"VLM 直播间 GUI")
  print(f"  视频: {runtime['video_path']} ({runtime['player'].duration:.1f}s)")
  print(f"  弹幕: {runtime['danmaku_path'] or '无'}")
  print(f"  人设: {args.persona}  模型: {args.model}")
  print(f"  端口: {args.port}")

  @ui.page("/")
  async def vlm_page():
    page_client = context.client
    # ── 页面局部状态 ──
    streaming_labels: dict[str, ui.label] = {}
    streamed_ids: collections.deque[str] = collections.deque(maxlen=50)
    terminal_state = {"active_response_id": None}
    danmaku_entries: list[ui.element] = []
    callback_refs = {
      "response_fn": None,
      "chunk_fn": None,
      "pre_fn": None,
      "danmaku_fn": None,
      "stop_fn": None,
      "bound_player": None,
      "bound_studio": None,
    }

    DANMAKU_HEIGHT = "height: calc(55vh - 120px)"
    CONSOLE_HEIGHT = "height: calc(35vh - 50px)"

    def _current_player() -> VideoPlayer:
      return runtime["player"]

    def _current_studio() -> StreamingStudio:
      return runtime["studio"]

    def _list_local_files(suffixes: tuple[str, ...]) -> list[str]:
      """列出 data/ 下可选文件（返回项目相对路径）"""
      data_dir = project_root / "data"
      if not data_dir.exists():
        return []
      items: list[str] = []
      suffix_set = {s.lower() for s in suffixes}
      for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffix_set:
          try:
            items.append(str(path.relative_to(project_root)))
          except ValueError:
            items.append(str(path))
      items.sort()
      return items

    def _open_source_picker(
      title: str,
      suffixes: tuple[str, ...],
      target_input: ui.input,
    ) -> None:
      """打开本地文件选择对话框（用于视频/弹幕换源）"""
      options = _list_local_files(suffixes)
      if not options:
        ui.notify("data/ 目录下没有可选文件，请先下载数据", type="warning")
        return

      dialog = ui.dialog()
      with dialog, ui.card().classes("w-[56rem] max-w-[95vw]"):
        ui.label(title).classes("text-base font-bold")
        picker = ui.select(
          options=options,
          value=options[0],
        ).props("dense outlined use-input fill-input")
        picker.classes("w-full")

        with ui.row().classes("w-full justify-end gap-2"):
          ui.button("取消", on_click=dialog.close).props("flat")

          def _confirm():
            if picker.value:
              target_input.value = str(picker.value)
            dialog.close()

          ui.button("选择", on_click=_confirm).props("color=primary")

      dialog.open()

    # ── 顶部控制栏 ──
    with ui.row().classes(
      "w-full items-center gap-4 px-4 py-2 bg-gray-100 shrink-0"
    ):
      ui.label("VLM 直播间").classes("text-xl font-bold")

      async def on_start():
        if _current_studio().is_running:
          return
        try:
          print("[GUI] 开始按钮点击，正在启动...", flush=True)
          start_btn.disable()
          runtime["gui_chunk_queue"].clear()
          runtime["gui_pre_response_queue"].clear()
          runtime["gui_response_queue"].clear()
          runtime["gui_danmaku_queue"].clear()
          _mjpeg_state["jpeg_bytes"] = b""
          _mjpeg_state["running"] = True
          _reload_video_stream()
          await _current_studio().start()
          stop_btn.enable()
          print("[GUI] 启动完成", flush=True)
        except Exception as e:
          print(f"[GUI] 启动失败: {e}", flush=True)
          import traceback; traceback.print_exc()
          _mjpeg_state["running"] = False
          start_btn.enable()
          ui.notify(f"启动失败: {e}", type="negative")

      async def on_stop():
        if not _current_studio().is_running:
          stop_btn.disable()
          start_btn.enable()
          return
        stop_btn.disable()
        _mjpeg_state["running"] = False
        _cleanup_callbacks(_force=True)
        await _current_studio().stop()
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
        f"0.0 / {_current_player().duration:.1f}s"
      ).classes("text-sm text-gray-600 whitespace-nowrap")

    with ui.row().classes(
      "w-full items-center gap-2 px-4 pb-2 bg-gray-100 shrink-0"
    ):
      with ui.row().classes("flex-1 items-center gap-2"):
        video_input = ui.input(label="视频路径").props("dense outlined").classes("flex-1")
        ui.button(
          "选择视频",
          on_click=lambda: _open_source_picker(
            "选择视频文件",
            (".mp4", ".mkv", ".flv", ".webm"),
            video_input,
          ),
        ).props("dense")
      video_input.value = runtime["video_path"]

      with ui.row().classes("flex-1 items-center gap-2"):
        danmaku_input = ui.input(
          label="弹幕XML路径（可空）"
        ).props("dense outlined").classes("flex-1")
        ui.button(
          "选择弹幕",
          on_click=lambda: _open_source_picker(
            "选择弹幕 XML 文件",
            (".xml",),
            danmaku_input,
          ),
        ).props("dense")
      danmaku_input.value = runtime["danmaku_path"] or ""

      async def on_switch_source():
        raw_video = (video_input.value or "").strip()
        raw_danmaku = (danmaku_input.value or "").strip()

        if not raw_video:
          ui.notify("请先填写视频文件路径", type="warning")
          return

        normalized_video = _normalize_path(raw_video)
        if not Path(normalized_video).exists():
          ui.notify(f"视频文件不存在: {normalized_video}", type="negative")
          return

        normalized_danmaku = _normalize_path(raw_danmaku) if raw_danmaku else None
        if normalized_danmaku and not Path(normalized_danmaku).exists():
          ui.notify(f"弹幕文件不存在: {normalized_danmaku}", type="negative")
          return

        if (
          normalized_video == runtime["video_path"]
          and normalized_danmaku == runtime["danmaku_path"]
        ):
          ui.notify("视频与弹幕源未发生变化", type="info")
          return

        switch_btn.disable()
        start_btn.disable()
        stop_btn.disable()
        _mjpeg_state["running"] = False
        _mjpeg_state["jpeg_bytes"] = b""

        old_studio = _current_studio()
        try:
          _cleanup_callbacks(_force=True)
          if old_studio.is_running:
            await old_studio.stop()

          # 切换源视为“重启新会话”：清理运行期状态，避免记忆串场
          old_studio.llm_wrapper.clear_history()
          old_studio._comment_buffer.clear()
          old_studio._response_queue = asyncio.Queue()
          old_studio._last_collect_time = None
          old_studio._last_reply_time = None
          old_studio._last_prompt = None
          old_studio._prev_scene_description = None
          old_studio._pending_comment_count = 0
          old_studio._current_frame_b64 = None
          old_studio._last_used_timing = None

          mem_mgr = old_studio.llm_wrapper.memory_manager
          if mem_mgr is not None:
            mem_mgr.clear_runtime_state(clear_summary=True)

          _clear_ui_state()

          new_runtime = _build_runtime(normalized_video, normalized_danmaku)
          runtime.update(new_runtime)
          _reload_video_stream()

          video_input.value = runtime["video_path"]
          danmaku_input.value = runtime["danmaku_path"] or ""

          progress.set_value(0)
          time_label.set_text(f"0.0 / {_current_player().duration:.1f}s")
          _add_console_block(
            (
              f"[系统] 已切换视频源（会话已重置）\n"
              f"视频: {runtime['video_path']}\n"
              f"弹幕: {runtime['danmaku_path'] or '无'}"
            ),
            "text-sm text-cyan-300 font-mono whitespace-pre-wrap",
          )
          ui.notify("已切换视频/弹幕源", type="positive")
        except Exception as e:
          ui.notify(f"切换失败: {e}", type="negative")
          _add_console_block(
            f"[系统] 切换失败: {e}",
            "text-sm text-red-300 font-mono whitespace-pre-wrap",
          )
        finally:
          switch_btn.enable()
          start_btn.enable()
          stop_btn.disable()

      switch_btn = ui.button("切换源", on_click=on_switch_source).props(
        "dense icon=sync color=primary"
      )

    # ── 中间区域：视频画面 + 弹幕侧栏 ──
    with ui.row().classes("w-full gap-0 px-2").style("height: 55vh"):

      # 左侧：视频画面（MJPEG 流，浏览器原生支持无闪烁连续帧）
      with ui.column().classes("h-full p-1 overflow-hidden").style("flex: 3"):
        ui.html(
          '<img id="mjpeg-video" src="/video-stream" '
          'style="width:100%;height:100%;object-fit:contain;background:#111;display:block;" />',
          sanitize=False,
        ).classes("w-full h-full overflow-hidden")

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
            if not content or not _current_studio().is_running:
              return
            comment = Comment(
              user_id="manual_user",
              nickname="手动观众",
              content=content,
            )
            _current_studio().send_comment(comment)
            _add_danmaku("手动观众", content, priority=True)
            msg_input.value = ""

          ui.button("发送", on_click=send_comment).props("dense")
          msg_input.on("keydown.enter", send_comment)

    def _reload_video_stream():
      """强制刷新 MJPEG 连接，解决切换源后浏览器复用旧连接导致卡住。"""
      try:
        ui.run_javascript(
          'const img = document.getElementById("mjpeg-video");'
          'if (img) img.src = "/video-stream?" + Date.now();'
        )
      except Exception as e:
        print(f"[MJPEG] 刷新视频流失败: {e}", flush=True)

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

    def _clear_ui_state() -> None:
      """清空页面状态（用于切源时模拟重启）"""
      for entry in danmaku_entries:
        try:
          entry.delete()
        except Exception:
          pass
      danmaku_entries.clear()

      for lbl in streaming_labels.values():
        try:
          lbl.delete()
        except Exception:
          pass
      streaming_labels.clear()

      for lbl in gui_streaming_labels.values():
        try:
          lbl.delete()
        except Exception:
          pass
      gui_streaming_labels.clear()

      streamed_ids.clear()
      gui_streamed_ids.clear()
      terminal_state["active_response_id"] = None
      runtime["gui_chunk_queue"].clear()
      runtime["gui_pre_response_queue"].clear()
      runtime["gui_response_queue"].clear()
      runtime["gui_danmaku_queue"].clear()

    # ── 回调函数 ──

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
      active_id = terminal_state["active_response_id"]
      if chunk.response_id != active_id:
        terminal_state["active_response_id"] = chunk.response_id
      if chunk.done:
        print(f"[主播] {chunk.accumulated}")
        terminal_state["active_response_id"] = None

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
      print(f"[主播] {response.content}")
      _add_console_block(
        response.content,
        "text-sm text-gray-200 font-mono whitespace-pre-wrap",
      )

    # ── 回调注册 / 清理 ──

    def _register_callbacks():
      _cleanup_callbacks()
      current_player = _current_player()
      current_studio = _current_studio()
      print(f"[CB] 注册回调: chunk_cbs_before={len(current_studio._chunk_callbacks)}", flush=True)

      callback_refs["danmaku_fn"] = on_danmaku
      callback_refs["pre_fn"] = on_pre_response
      callback_refs["chunk_fn"] = on_chunk
      callback_refs["response_fn"] = on_response
      callback_refs["bound_player"] = current_player
      callback_refs["bound_studio"] = current_studio

      def on_auto_stop():
        """视频播完由 studio 回调触发，在 GUI 线程中安排清理"""
        async def _do_stop():
          if callback_refs.get("bound_studio") is not current_studio:
            return
          stop_btn.disable()
          _mjpeg_state["running"] = False
          _cleanup_callbacks(_force=True)
          await current_studio.stop()
          start_btn.enable()
          progress.set_value(1.0)
          total = current_player.duration
          time_label.set_text(f"{total:.1f} / {total:.1f}s (完毕)")
        ui.timer(0.1, lambda: asyncio.ensure_future(_do_stop()), once=True)

      callback_refs["stop_fn"] = on_auto_stop

      current_player.on_danmaku(on_danmaku)
      current_studio.on_pre_response(on_pre_response)
      current_studio.on_response_chunk(on_chunk)
      current_studio.on_response(on_response)
      current_studio.on_stop(on_auto_stop)
      print(f"[CB] 注册完成: chunk_cbs_after={len(current_studio._chunk_callbacks)}", flush=True)

    def _cleanup_callbacks(_force=False):
      bound_studio = callback_refs.get("bound_studio")
      if not _force and bound_studio and bound_studio.is_running:
        return
      bound_player = callback_refs.get("bound_player")

      fn = callback_refs.get("danmaku_fn")
      if fn and bound_player and fn in bound_player._on_danmaku_callbacks:
        bound_player._on_danmaku_callbacks.remove(fn)
      callback_refs["danmaku_fn"] = None

      fn = callback_refs.get("pre_fn")
      if fn and bound_studio and fn in bound_studio._pre_response_callbacks:
        bound_studio._pre_response_callbacks.remove(fn)
      callback_refs["pre_fn"] = None

      fn = callback_refs.get("chunk_fn")
      if fn and bound_studio:
        bound_studio.remove_chunk_callback(fn)
      callback_refs["chunk_fn"] = None

      fn = callback_refs.get("response_fn")
      if fn and bound_studio:
        bound_studio.remove_callback(fn)
      callback_refs["response_fn"] = None

      fn = callback_refs.get("stop_fn")
      if fn and bound_studio:
        bound_studio.remove_stop_callback(fn)
      callback_refs["stop_fn"] = None

      callback_refs["bound_player"] = None
      callback_refs["bound_studio"] = None
      streaming_labels.clear()

    context.client.on_disconnect(_cleanup_callbacks)

    # ── 进度条定时器 ──

    def update_progress():
      player = _current_player()
      studio = _current_studio()
      if not studio.is_running:
        _mjpeg_state["running"] = False
        stop_btn.disable()
        start_btn.enable()
        return
      current = player.current_sec
      total = player.duration
      if total > 0:
        progress.set_value(current / total)
        time_label.set_text(f"{current:.1f} / {total:.1f}s")
      if player.is_finished:
        time_label.set_text(f"{total:.1f} / {total:.1f}s (完毕)")

    ui.timer(0.5, update_progress)

    gui_streaming_labels: dict = {}
    gui_streamed_ids: list = []

    def _poll_gui_queues():
      """Poll shared queues and update GUI elements (immune to NiceGUI disconnect)"""
      if getattr(page_client, "_deleted", False) or not page_client.has_socket_connection:
        return
      chunk_q = runtime.get("gui_chunk_queue")
      pre_q = runtime.get("gui_pre_response_queue")
      resp_q = runtime.get("gui_response_queue")
      dm_q = runtime.get("gui_danmaku_queue")

      while dm_q:
        danmaku = dm_q.popleft()
        nick = f"观众{danmaku.user_hash[:4]}" if danmaku.user_hash else "观众"
        _add_danmaku(nick, danmaku.content)

      while pre_q:
        old_comments, new_comments = pre_q.popleft()
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

      while chunk_q:
        chunk = chunk_q.popleft()
        if chunk.response_id not in gui_streaming_labels:
          with console_column:
            lbl = ui.label(chunk.accumulated).classes(
              "text-sm text-green-400 font-mono whitespace-pre-wrap"
            )
          gui_streaming_labels[chunk.response_id] = lbl
          console_scroll.scroll_to(percent=1.0)
        else:
          lbl = gui_streaming_labels[chunk.response_id]
          lbl.set_text(chunk.accumulated)
          console_scroll.scroll_to(percent=1.0)
        if chunk.done:
          lbl = gui_streaming_labels.pop(chunk.response_id, None)
          if lbl is not None:
            lbl.classes(replace="text-sm text-gray-200 font-mono whitespace-pre-wrap")
          gui_streamed_ids.append(chunk.response_id)

      while resp_q:
        response = resp_q.popleft()
        # 兜底：即使 chunk.done 丢失，也在完整回复到达时把对应绿色行收尾成灰白
        active_lbl = gui_streaming_labels.pop(response.id, None)
        if active_lbl is not None:
          active_lbl.set_text(response.content)
          active_lbl.classes(replace="text-sm text-gray-200 font-mono whitespace-pre-wrap")
          gui_streamed_ids.append(response.id)
          continue
        if response.id in gui_streamed_ids:
          continue
        _add_console_block(
          response.content,
          "text-sm text-gray-200 font-mono whitespace-pre-wrap",
        )

    ui.timer(0.2, _poll_gui_queues)

  if sys.platform == "win32":
    _default_handler = None

    def _quiet_exception_handler(loop, context):
      exc = context.get("exception")
      if isinstance(exc, ConnectionResetError):
        return
      if _default_handler:
        _default_handler(context)
      else:
        loop.default_exception_handler(context)

    @app.on_startup
    async def _patch_event_loop():
      nonlocal _default_handler
      loop = asyncio.get_running_loop()
      _default_handler = loop.get_exception_handler()
      loop.set_exception_handler(_quiet_exception_handler)

  ui.run(port=args.port, reload=False, title="VLM 直播间")


if __name__ == "__main__":
  main()
