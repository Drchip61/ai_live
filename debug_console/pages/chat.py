"""
模拟直播间页面
左栏：主播发言  右栏：观众弹幕 + 输入
支持单用户和多用户（随机身份）两种模式
"""

import collections
from datetime import datetime

from coolname import generate
from nicegui import ui, context

from debug_console.auto_viewer import AutoViewer
from streaming_studio import StreamingStudio, Comment
from streaming_studio.models import ResponseChunk


def _random_identity() -> tuple[str, str]:
  """
  生成随机用户身份

  Returns:
    (user_id, nickname)
  """
  words = generate(2)
  user_id = "_".join(w.lower() for w in words)
  nickname = " ".join(w.capitalize() for w in words)
  return user_id, nickname


def create_chat_page(studio: StreamingStudio) -> None:
  """
  构建模拟直播间 UI（左右分栏）

  Args:
    studio: 直播间实例
  """
  # 页面局部状态
  state = {
    "multi_user": True,
    "user_id": "test_user",
    "nickname": "测试用户",
    "next_id": "",
    "next_nick": "",
  }
  state["next_id"], state["next_nick"] = _random_identity()

  # 自动观众引擎
  auto_viewer = AutoViewer(studio)

  # 回调引用（stop / disconnect 时移除）
  callback_ref = {"fn": None, "chunk_fn": None}

  # 流式气泡追踪 {response_id: (content_label, card)}
  streaming_bubbles: dict[str, tuple[ui.label, ui.card]] = {}
  # 已完成流式输出的 response_id（避免 on_response 重复，有界防泄漏）
  streamed_ids: collections.deque[str] = collections.deque(maxlen=50)

  PANEL_HEIGHT = "height: calc(100vh - 350px); min-height: 300px"

  with ui.column().classes("w-full h-full p-4 gap-4"):
    # ── 控制栏 ──
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
      ui.label("模拟直播间").classes("text-2xl font-bold")

      status_label = ui.label().classes("text-sm")

      ui.space()

      async def on_start():
        await studio.start()
        status_label.text = "运行中"
        start_btn.disable()
        stop_btn.enable()

      async def on_stop():
        if auto_viewer.is_running:
          await auto_viewer.stop()
          auto_switch.value = False
          auto_label.text = "已停止"
          auto_label.classes(replace="text-xs text-gray-400")
        if callback_ref["fn"]:
          studio.remove_callback(callback_ref["fn"])
          callback_ref["fn"] = None
        if callback_ref["chunk_fn"]:
          studio.remove_chunk_callback(callback_ref["chunk_fn"])
          callback_ref["chunk_fn"] = None
        streaming_bubbles.clear()
        await studio.stop()
        status_label.text = "已停止"
        start_btn.enable()
        stop_btn.disable()

      start_btn = ui.button("启动", on_click=on_start).props("dense")
      stop_btn = ui.button("停止", on_click=on_stop).props("dense")
      stop_btn.disable()

      ui.separator().props("vertical")

      async def on_auto_toggle(e):
        if e.value:
          await auto_viewer.start()
          auto_label.text = "运行中"
          auto_label.classes(replace="text-xs text-green-600")
        else:
          await auto_viewer.stop()
          auto_label.text = "已停止"
          auto_label.classes(replace="text-xs text-gray-400")

      auto_switch = ui.switch("自动观众", on_change=on_auto_toggle).props("dense")
      auto_label = ui.label("已停止").classes("text-xs text-gray-400")

    # ── 用户模式切换 ──
    with ui.row().classes("w-full items-center gap-4 flex-wrap"):
      def on_mode_change(e):
        is_multi = e.value == "multi"
        state["multi_user"] = is_multi
        single_inputs.set_visibility(not is_multi)
        identity_preview.set_visibility(is_multi)
        if is_multi:
          state["next_id"], state["next_nick"] = _random_identity()
          preview_label.text = (
            f"下一个身份: {state['next_nick']} ({state['next_id']})"
          )

      ui.toggle(
        {"single": "单用户", "multi": "多用户（随机）"},
        value="multi",
        on_change=on_mode_change,
      ).props("dense")

      single_inputs = ui.row().classes("gap-2")
      single_inputs.set_visibility(False)
      with single_inputs:
        uid_input = (
          ui.input("用户ID", value="test_user").props("dense").classes("w-32")
        )
        nick_input = (
          ui.input("昵称", value="测试用户").props("dense").classes("w-32")
        )

        def on_uid_change(e):
          state["user_id"] = e.args
        def on_nick_change(e):
          state["nickname"] = e.args
        uid_input.on("update:model-value", on_uid_change)
        nick_input.on("update:model-value", on_nick_change)

      identity_preview = ui.row()
      with identity_preview:
        preview_label = ui.label(
          f"下一个身份: {state['next_nick']} ({state['next_id']})"
        ).classes("text-sm text-gray-500")

    ui.separator()

    # ── 左右分栏 ──
    with ui.row().classes("w-full flex-1 gap-4"):

      # ── 左栏：主播发言 ──
      with ui.column().classes("flex-1 gap-2"):
        ui.label("主播发言").classes("text-sm font-bold text-blue-700")
        with ui.scroll_area().classes(
          "w-full bg-blue-50 rounded border"
        ).style(PANEL_HEIGHT) as response_scroll:
          response_container = ui.column().classes("w-full gap-2 p-2")

      # ── 右栏：观众弹幕 + 输入 ──
      with ui.column().classes("flex-1 gap-2"):
        ui.label("观众弹幕").classes("text-sm font-bold text-green-700")
        with ui.scroll_area().classes(
          "w-full bg-gray-50 rounded border"
        ).style(PANEL_HEIGHT) as comment_scroll:
          comment_container = ui.column().classes("w-full gap-2 p-2")

        # 输入栏（在右栏底部）
        with ui.row().classes("w-full gap-2"):
          msg_input = ui.input(placeholder="输入弹幕...").props(
            "dense outlined"
          ).classes("flex-1")

          def send():
            content = msg_input.value.strip()
            if not content or not studio.is_running:
              return

            if state["multi_user"]:
              user_id = state["next_id"]
              nickname = state["next_nick"]
            else:
              user_id = state["user_id"] or "test_user"
              nickname = state["nickname"] or "测试用户"

            now = datetime.now()
            comment = Comment(
              user_id=user_id,
              nickname=nickname,
              content=content,
            )
            studio.send_comment(comment)
            _add_comment_bubble(nickname, content, now.strftime("%H:%M:%S"))

            msg_input.value = ""

            if state["multi_user"]:
              state["next_id"], state["next_nick"] = _random_identity()
              preview_label.text = (
                f"下一个身份: {state['next_nick']} ({state['next_id']})"
              )

          ui.button("发送", on_click=send).props("dense")
          msg_input.on("keydown.enter", send)

  # ── 辅助函数 ──

  def _add_comment_bubble(nickname: str, content: str, timestamp: str):
    """添加弹幕到右栏"""
    with comment_container:
      with ui.card().classes("w-full px-3 py-1"):
        with ui.row().classes("items-center gap-2"):
          ui.label(nickname).classes("text-xs font-bold text-green-700")
          ui.label(timestamp).classes("text-xs text-gray-400")
        ui.label(content).classes("text-sm")
    comment_scroll.scroll_to(percent=1.0)

  def _add_response_bubble(content: str):
    """添加主播回复到左栏"""
    with response_container:
      with ui.card().classes("w-full px-3 py-2 bg-white"):
        ui.label(content).classes("text-sm")
        ui.label(datetime.now().strftime("%H:%M:%S")).classes(
          "text-xs text-gray-400"
        )
    response_scroll.scroll_to(percent=1.0)

  # ── 回调注册 ──

  def on_response(response):
    """完整回复回调（非流式，或流式的兜底）"""
    if response.id in streamed_ids:
      return
    _add_response_bubble(response.content)

  callback_ref["fn"] = on_response
  studio.on_response(on_response)

  def on_chunk(chunk: ResponseChunk):
    """流式回复片段回调"""
    if chunk.response_id not in streaming_bubbles:
      # 首个 chunk：在左栏创建可更新的气泡
      with response_container:
        card = ui.card().classes("w-full px-3 py-2 bg-white")
        with card:
          content_label = ui.label(chunk.accumulated).classes("text-sm")
      streaming_bubbles[chunk.response_id] = (content_label, card)
      response_scroll.scroll_to(percent=1.0)
    else:
      content_label, card = streaming_bubbles[chunk.response_id]
      content_label.set_text(chunk.accumulated)
      response_scroll.scroll_to(percent=1.0)

    if chunk.done:
      # 添加时间戳（与非流式气泡一致）
      _, card = streaming_bubbles.pop(chunk.response_id, (None, None))
      if card is not None:
        with card:
          ui.label(datetime.now().strftime("%H:%M:%S")).classes(
            "text-xs text-gray-400"
          )
      streamed_ids.append(chunk.response_id)
      response_scroll.scroll_to(percent=1.0)

  callback_ref["chunk_fn"] = on_chunk
  studio.on_response_chunk(on_chunk)

  # ── 自动观众弹幕显示回调 ──

  def on_auto_comment(comment: Comment):
    """自动观众弹幕 → 显示到右栏"""
    _add_comment_bubble(
      comment.nickname,
      comment.content,
      comment.timestamp.strftime("%H:%M:%S"),
    )

  auto_viewer.on_comment(on_auto_comment)

  # ── 清理 ──

  def cleanup():
    auto_viewer.remove_comment_callback(on_auto_comment)
    if auto_viewer.is_running:
      import asyncio
      asyncio.create_task(auto_viewer.stop())
    if callback_ref["fn"]:
      studio.remove_callback(callback_ref["fn"])
      callback_ref["fn"] = None
    if callback_ref["chunk_fn"]:
      studio.remove_chunk_callback(callback_ref["chunk_fn"])
      callback_ref["chunk_fn"] = None

  context.client.on_disconnect(cleanup)
