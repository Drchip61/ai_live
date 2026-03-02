"""
运行时调试设置页面
提供热切换参数的 NiceGUI 控件

直播间基础参数始终显示；若 studio 关联了 VideoPlayer，
则额外显示 VLM 图像参数区块。
"""

from nicegui import ui

from streaming_studio import StreamingStudio


def create_settings_page(studio: StreamingStudio) -> None:
  """
  构建运行时调试设置页面。

  Args:
    studio: 直播间实例（必须）
  """
  player = studio._video_player  # None = 纯文本模式；有值 = VLM 模式

  with ui.column().classes("w-full gap-4 p-4"):
    ui.label("运行时调试设置").classes("text-2xl font-bold")
    ui.label("以下参数修改后立即生效，无需重启。").classes("text-sm text-gray-500")

    # ── 直播间基础参数 ──────────────────────────────────────────────
    with ui.card().classes("w-full"):
      ui.label("直播间参数").classes("text-lg font-bold")
      ui.separator()
      with ui.grid(columns=2).classes("w-full gap-x-8 gap-y-4 pt-2"):

        with ui.column():
          ui.label("最小回复间隔 (s)").classes("text-sm text-gray-600")
          ui.number(
            value=studio.min_interval,
            min=0.5, max=60, step=0.5, format="%.1f",
            on_change=lambda e: setattr(studio, "min_interval", e.value),
          ).classes("w-full")

        with ui.column():
          ui.label("最大回复间隔 (s)").classes("text-sm text-gray-600")
          ui.number(
            value=studio.max_interval,
            min=1, max=120, step=0.5, format="%.1f",
            on_change=lambda e: setattr(studio, "max_interval", e.value),
          ).classes("w-full")

        with ui.column():
          ui.label("每次回复收集弹幕数上限").classes("text-sm text-gray-600")
          ui.number(
            value=studio.recent_comments_limit,
            min=1, max=100, step=1, format="%d",
            on_change=lambda e: setattr(studio, "recent_comments_limit", int(e.value)),
          ).classes("w-full")

        with ui.column().classes("justify-end pb-1"):
          ui.label("流式输出").classes("text-sm text-gray-600")
          ui.switch(
            value=studio.enable_streaming,
            on_change=lambda e: setattr(studio, "enable_streaming", e.value),
          )

    # ── VLM 图像参数（仅 studio 关联了 VideoPlayer 时显示）─────────
    if player is not None:
      from streaming_studio.studio import VlmMode

      with ui.card().classes("w-full"):
        ui.label("VLM 图像参数").classes("text-lg font-bold")
        ui.separator()
        with ui.grid(columns=2).classes("w-full gap-x-8 gap-y-4 pt-2"):

          with ui.column():
            ui.label("VLM 模式").classes("text-sm text-gray-600")
            ui.select(
              options={
                "two_pass":        "two_pass — 两趟（默认）",
                "direct":          "direct — 直接输图",
                "summary_only":    "summary_only — 仅文字总结",
                "two_pass_cached": "two_pass_cached — 两趟+缓存",
              },
              value=studio._vlm_mode.value,
              on_change=lambda e: setattr(studio, "_vlm_mode", VlmMode(e.value)),
            ).classes("w-full")

          with ui.column():
            ui.label("播放速度倍率").classes("text-sm text-gray-600")
            ui.number(
              value=player.speed,
              min=0.1, max=10, step=0.1, format="%.1f",
              on_change=lambda e: setattr(player, "speed", e.value),
            ).classes("w-full")

          with ui.column():
            ui.label("帧采样间隔 (s)").classes("text-sm text-gray-600")
            ui.number(
              value=player.frame_interval,
              min=1, max=60, step=0.5, format="%.1f",
              on_change=lambda e: setattr(player, "frame_interval", e.value),
            ).classes("w-full")
