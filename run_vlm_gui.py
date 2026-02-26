"""
VLM 直播间 GUI
NiceGUI Web 界面，集成视频画面、弹幕侧栏、用户输入和控制台输出

用法:
  python run_vlm_gui.py
  python run_vlm_gui.py --video <视频文件> --danmaku <弹幕XML>

示例:
  python run_vlm_gui.py
  python run_vlm_gui.py --video data/sample.mp4 --danmaku data/sample.xml
  python run_vlm_gui.py --video data/sample.mp4 --persona kuro --model anthropic
  python run_vlm_gui.py --lan --port 8081
"""

import argparse
import asyncio
import base64
import collections
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from nicegui import ui, context, app
from starlette.responses import StreamingResponse

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio, Comment
from streaming_studio.models import ResponseChunk
from video_source import VideoPlayer
from debug_console.state_collector import StateCollector
from debug_console.comment_broadcaster import CommentBroadcaster
from debug_console.pages.monitor import create_monitor_page


def parse_args():
  parser = argparse.ArgumentParser(
    description="VLM 直播间 GUI — 视频画面 + 弹幕 → 多模态理解（NiceGUI 版）",
  )
  parser.add_argument(
    "--video", default=None,
    help="视频文件路径（可选，可在 GUI 中选择）",
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
    "--lan", action="store_true", default=False,
    help="在局域网中部署（绑定 0.0.0.0，允许其它设备访问）",
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
    _mjpeg_state["jpeg_bytes"] = base64.b64decode(frame.base64_jpeg)

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
    player.on_display_frame(_store_display_frame)
    return {
      "video_path": normalized_video,
      "danmaku_path": normalized_danmaku,
      "player": player,
      "studio": studio,
    }

  if args.video:
    runtime = _build_runtime(args.video, args.danmaku)
  else:
    runtime = {
      "video_path": None,
      "danmaku_path": None,
      "player": None,
      "studio": None,
    }

  async def _mjpeg_generator():
    while True:
      player = runtime.get("player")
      if _mjpeg_state["running"] and player:
        interval = 1.0 / max(player.display_fps, 1)
        data = _mjpeg_state["jpeg_bytes"]
        if data:
          yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + data
            + b"\r\n"
          )
          await asyncio.sleep(interval)
          continue
      await asyncio.sleep(0.2)

  @app.get("/video-stream")
  async def video_stream_route():
    return StreamingResponse(
      _mjpeg_generator(),
      media_type="multipart/x-mixed-replace; boundary=frame",
    )

  host = "0.0.0.0" if args.lan else "127.0.0.1"
  broadcaster = CommentBroadcaster()

  print(f"VLM 直播间 GUI")
  if runtime.get("player"):
    print(f"  视频: {runtime['video_path']} ({runtime['player'].duration:.1f}s)")
    print(f"  弹幕: {runtime['danmaku_path'] or '无'}")
  else:
    print(f"  视频: 未指定（请在 GUI 中选择）")
  print(f"  人设: {args.persona}  模型: {args.model}")
  print(f"  地址: {host}:{args.port}")

  @ui.page("/monitor")
  def monitor_page():
    studio = runtime.get("studio")
    if not studio:
      with ui.column().classes("w-full items-center justify-center p-8"):
        ui.label("直播间未初始化").classes("text-xl text-gray-500")
        ui.label("请先在主页选择视频并启动").classes("text-gray-400")
      return
    collector = StateCollector(studio)
    create_monitor_page(collector)

  @ui.page("/")
  async def vlm_page():
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
        studio = runtime.get("studio")
        if not studio:
          ui.notify("请先选择视频文件并点击「切换源」", type="warning")
          return
        if studio.is_running:
          return
        start_btn.disable()
        _mjpeg_state["running"] = True
        _register_callbacks()
        await _current_studio().start()
        stop_btn.enable()

      async def on_stop():
        if not _current_studio().is_running:
          stop_btn.disable()
          start_btn.enable()
          return
        stop_btn.disable()
        _mjpeg_state["running"] = False
        _cleanup_callbacks()
        await _current_studio().stop()
        start_btn.enable()
        progress.set_value(0)

      start_btn = ui.button("开始", on_click=on_start).props(
        "dense icon=play_arrow color=green"
      )
      if not runtime.get("studio"):
        start_btn.disable()
      stop_btn = ui.button("停止", on_click=on_stop).props(
        "dense icon=stop color=red"
      )
      stop_btn.disable()

      ui.link("监控面板", "/monitor", new_tab=True).classes(
        "text-sm text-blue-600 no-underline hover:underline"
      )

      progress = ui.linear_progress(value=0, show_value=False).classes("flex-1")
      _init_player = runtime.get("player")
      time_label = ui.label(
        f"0.0 / {_init_player.duration:.1f}s" if _init_player else "未加载视频"
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
      video_input.value = runtime["video_path"] or ""

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
          runtime.get("video_path")
          and normalized_video == runtime["video_path"]
          and normalized_danmaku == runtime["danmaku_path"]
        ):
          ui.notify("视频与弹幕源未发生变化", type="info")
          return

        switch_btn.disable()
        start_btn.disable()
        stop_btn.disable()
        _mjpeg_state["running"] = False
        _mjpeg_state["jpeg_bytes"] = b""

        old_studio = runtime.get("studio")
        try:
          _cleanup_callbacks()
          if old_studio:
            if old_studio.is_running:
              await old_studio.stop()

            # 切换源视为"重启新会话"：清理运行期状态，避免记忆串场
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
          if runtime.get("studio"):
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
          '<img src="/video-stream" '
          'style="width:100%;height:100%;object-fit:contain;background:#111;">',
          sanitize=False,
        ).classes("w-full h-full")

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
            broadcaster.broadcast(comment)
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

      streamed_ids.clear()
      terminal_state["active_response_id"] = None

    # ── 回调函数 ──

    def on_danmaku(danmaku):
      """视频弹幕到达 → 广播给所有客户端"""
      nick = (
        f"观众{danmaku.user_hash[:4]}"
        if danmaku.user_hash
        else "观众"
      )
      comment = Comment(
        user_id=danmaku.user_hash or "anonymous",
        nickname=nick,
        content=danmaku.content,
      )
      broadcaster.broadcast(comment)

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
      # 同步输出到终端，便于无界面排障与远程观察
      active_id = terminal_state["active_response_id"]
      if chunk.response_id != active_id:
        if active_id is not None:
          print()
        terminal_state["active_response_id"] = chunk.response_id
        print("[主播] ", end="", flush=True)
      if chunk.chunk:
        print(chunk.chunk, end="", flush=True)
      if chunk.done:
        print()
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
          _cleanup_callbacks()
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

    def _cleanup_callbacks():
      bound_player = callback_refs.get("bound_player")
      bound_studio = callback_refs.get("bound_studio")

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

    def _on_broadcast(comment: Comment):
      """broadcaster 回调 → 本客户端弹幕侧栏显示"""
      _add_danmaku(comment.nickname, comment.content, priority=comment.priority)

    broadcaster.register(_on_broadcast)

    def _on_disconnect():
      _cleanup_callbacks()
      broadcaster.unregister(_on_broadcast)

    context.client.on_disconnect(_on_disconnect)

    # ── 进度条定时器 ──

    def update_progress():
      player = runtime.get("player")
      studio = runtime.get("studio")
      if not player or not studio or not studio.is_running:
        return
      current = player.current_sec
      total = player.duration
      if total > 0:
        progress.set_value(current / total)
        time_label.set_text(f"{current:.1f} / {total:.1f}s")
      if player.is_finished:
        time_label.set_text(f"{total:.1f} / {total:.1f}s (完毕)")

    ui.timer(0.5, update_progress)

  ui.run(port=args.port, host=host, reload=False, title="VLM 直播间")


if __name__ == "__main__":
  main()
