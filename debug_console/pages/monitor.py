"""
监控面板页面
实时显示后台各模块运行状态
"""

from nicegui import ui

from debug_console.state_collector import StateCollector


def create_monitor_page(collector: StateCollector) -> None:
  """
  构建监控面板 UI

  Args:
    collector: 状态收集器
  """
  # 用于存放可刷新组件的容器引用
  containers = {}

  def refresh():
    """定时刷新回调"""
    try:
      state = collector.snapshot()
      _update_studio_card(containers.get("studio"), state.get("studio"))
      _update_timing_card(containers.get("timing"), state.get("timing"), state.get("studio"))
      _update_memory_card(containers.get("memory"), state.get("memory"))
      _update_topic_card(containers.get("topics"), state.get("topics"))
      _update_llm_card(containers.get("llm"), state.get("llm"))
      _update_prompt_card(containers.get("prompt"), state.get("studio"))
    except Exception:
      pass  # 刷新失败时静默跳过，等待下次重试

  with ui.column().classes("w-full gap-4 p-4"):
    # 标题栏
    with ui.row().classes("w-full items-center justify-between"):
      ui.label("实时监控").classes("text-2xl font-bold")
      ui.button("手动刷新", on_click=refresh).props("flat dense")

    # 直播间状态
    containers["studio"] = _build_studio_card()

    # 回复耗时分解（紧跟在直播间状态之后，便于一眼看到）
    containers["timing"] = _build_timing_card()

    # LLM 状态
    containers["llm"] = _build_llm_card()

    # 最近 prompt 展示
    containers["prompt"] = _build_prompt_card()

    # 话题管理器（放在 prompt 下方）
    containers["topics"] = _build_topic_card()

    # 记忆系统
    containers["memory"] = _build_memory_card()

  # 每 2 秒自动刷新
  ui.timer(2.0, refresh)


# ============================================================
# 卡片构建函数
# ============================================================

def _build_studio_card() -> dict:
  """构建直播间状态卡片，返回可更新的组件引用"""
  refs = {}
  with ui.card().classes("w-full"):
    ui.label("直播间状态").classes("text-lg font-bold")
    ui.separator()
    with ui.column().classes("gap-1"):
      refs["running"] = ui.label()
      refs["interval"] = ui.label()
      refs["buffer"] = ui.label()
      refs["pending"] = ui.label()
      refs["last_reply"] = ui.label()
      refs["totals"] = ui.label()
    ui.separator()
    ui.label("最近弹幕").classes("font-bold text-sm")
    refs["comments"] = ui.column().classes("gap-0 max-h-[200px] overflow-auto")
  return refs


def _build_memory_card() -> dict:
  """构建记忆系统卡片"""
  refs = {}
  with ui.card().classes("w-full"):
    ui.label("记忆系统").classes("text-lg font-bold")
    ui.separator()
    with ui.column().classes("gap-1"):
      refs["active_bar"] = ui.linear_progress(value=0, show_value=False).classes("w-full")
      refs["active_label"] = ui.label()
      refs["layers"] = ui.label()
      refs["interactions"] = ui.label()
      refs["tasks"] = ui.label()
    ui.separator()
    ui.label("Active 层内容").classes("font-bold text-sm")
    refs["active_list"] = ui.column().classes("gap-0 max-h-[200px] overflow-auto")
    ui.separator()
    with ui.expansion("Temporary 层内容", icon="schedule").classes("w-full"):
      refs["temporary_list"] = ui.column().classes("gap-0 max-h-[200px] overflow-auto")
    with ui.expansion("Summary 层内容", icon="summarize").classes("w-full"):
      refs["summary_list"] = ui.column().classes("gap-0 max-h-[200px] overflow-auto")
    with ui.expansion("Static 层内容", icon="push_pin").classes("w-full"):
      refs["static_list"] = ui.column().classes("gap-0 max-h-[200px] overflow-auto")
  return refs


def _build_topic_card() -> dict:
  """构建话题管理器卡片"""
  refs = {}
  with ui.card().classes("w-full"):
    ui.label("话题管理器").classes("text-lg font-bold")
    ui.separator()
    with ui.column().classes("gap-1"):
      refs["status"] = ui.label()
      refs["stats"] = ui.label()
      refs["timing"] = ui.label()
    ui.separator()
    ui.label("活跃话题").classes("font-bold text-sm")
    refs["topic_list"] = ui.column().classes("gap-1 max-h-[300px] overflow-auto")
  return refs


# 耗时卡片的阶段定义：(字段名, 显示名, Tailwind 颜色类)
_TIMING_STAGES = [
  ("comment_cluster_ms",  "弹幕聚类",       "text-gray-700"),
  ("reply_decision_ms",   "回复决策",       "text-gray-700"),
  ("topic_context_ms",    "话题上下文",     "text-gray-700"),
  ("prompt_format_ms",    "Prompt 格式化",  "text-gray-700"),
  ("scene_understand_ms", "场景理解(VLM)",  "text-blue-700"),
  ("memory_retrieval_ms", "记忆检索(RAG)",  "text-orange-700"),
  ("llm_first_token_ms",  "主回复-首token", "text-red-700"),
  ("llm_total_ms",        "主回复-总生成",  "text-red-700"),
  ("expression_map_ms",   "表情映射",       "text-gray-700"),
]

# 可选阶段：studio_state 中判断是否启用的字段名 → 未启用时的显示文字
_STAGE_ENABLED_BY: dict[str, tuple[str, str]] = {
  "comment_cluster_ms":  ("comment_clusterer_enabled", "未启用"),
  "topic_context_ms":    ("topic_manager_enabled",     "未启用"),
  "scene_understand_ms": ("vlm_mode",                  "非VLM模式"),
}


def _build_timing_card() -> dict:
  """构建回复耗时分解卡片"""
  refs = {}
  with ui.card().classes("w-full"):
    ui.label("回复耗时分解").classes("text-lg font-bold")
    ui.separator()
    refs["ts"] = ui.label("（等待首次回复）").classes("text-xs text-gray-400")
    refs["total"] = ui.label()

    # 为每个阶段预建一行：标签 + 进度条 + 数值
    refs["stages"] = {}
    for field_name, display_name, color_cls in _TIMING_STAGES:
      with ui.row().classes("w-full items-center gap-2"):
        ui.label(display_name).classes(f"text-xs w-28 shrink-0 {color_cls}")
        bar = ui.linear_progress(value=0, show_value=False).classes("flex-grow")
        value_label = ui.label("—").classes("text-xs w-20 text-right shrink-0")
      refs["stages"][field_name] = {"bar": bar, "label": value_label}
  return refs


def _build_llm_card() -> dict:
  """构建 LLM 状态卡片"""
  refs = {}
  with ui.card().classes("w-full"):
    ui.label("LLM 状态").classes("text-lg font-bold")
    ui.separator()
    with ui.column().classes("gap-1"):
      refs["model"] = ui.label()
      refs["persona"] = ui.label()
      refs["history"] = ui.label()
      refs["memory"] = ui.label()
      refs["bg_tasks"] = ui.label()
  return refs


def _build_prompt_card() -> dict:
  """构建最近 prompt 卡片"""
  refs = {}
  with ui.card().classes("w-full"):
    ui.label("最近一次发给模型的 Prompt").classes("text-lg font-bold")
    ui.separator()
    refs["prompt_text"] = ui.code("（等待首次触发）", language="markdown").classes(
      "w-full max-h-[300px] overflow-auto"
    )
  return refs


# ============================================================
# 卡片更新函数
# ============================================================

def _update_studio_card(refs: dict, state: dict) -> None:
  """更新直播间状态卡片"""
  if not refs or not state:
    return

  status = "运行中" if state["is_running"] else "已停止"
  refs["running"].text = f"状态: {status}"
  refs["interval"].text = (
    f"触发间隔: {state['min_interval']:.1f}s ~ {state['max_interval']:.1f}s"
  )
  refs["buffer"].text = (
    f"弹幕缓冲: {state['buffer_size']} / {state['buffer_max']}"
  )
  refs["pending"].text = f"待处理弹幕: {state['pending_comment_count']}"
  refs["last_reply"].text = (
    f"上次回复: {state['last_reply_time'] or '无'}"
  )
  refs["totals"].text = (
    f"累计弹幕: {state['total_comments']}  累计回复: {state['total_responses']}"
  )

  # 更新弹幕列表
  comments_container = refs["comments"]
  comments_container.clear()
  with comments_container:
    for c in reversed(state["recent_comments"]):
      ui.label(
        f"[{c['timestamp']}] {c['nickname']}: {c['content']}"
      ).classes("text-xs text-gray-600")


def _update_memory_card(refs: dict, state: dict) -> None:
  """更新记忆系统卡片"""
  if not refs:
    return

  if state is None:
    refs["active_label"].text = "记忆系统未启用"
    refs["layers"].text = ""
    refs["interactions"].text = ""
    refs["tasks"].text = ""
    refs["active_bar"].value = 0
    return

  capacity = state["active_capacity"]
  count = state["active_count"]
  refs["active_bar"].value = count / capacity if capacity > 0 else 0
  refs["active_label"].text = f"Active 层: {count} / {capacity}"
  refs["layers"].text = (
    f"Temporary: {state['temporary_count']}  "
    f"Summary: {state['summary_count']}  "
    f"Static: {state['static_count']}"
  )
  refs["interactions"].text = f"近期交互缓冲: {state['recent_interactions']}"

  summary_status = "运行中" if state["summary_task_running"] else "停止"
  cleanup_status = "运行中" if state["cleanup_task_running"] else "停止"
  refs["tasks"].text = f"汇总任务: {summary_status}  清理任务: {cleanup_status}"

  # 更新 active 层内容列表
  active_list = refs["active_list"]
  active_list.clear()
  with active_list:
    for m in state["active_memories"]:
      ui.label(
        f"[{m['timestamp']}] {m['content']}"
      ).classes("text-xs text-gray-600")

  # 更新 temporary 层内容列表
  temporary_list = refs["temporary_list"]
  temporary_list.clear()
  with temporary_list:
    for m in state.get("temporary_memories", []):
      sig = m.get("significance", 0)
      ui.label(
        f"[sig:{sig:.2f}] {m['content']}"
      ).classes("text-xs text-gray-600")
    if not state.get("temporary_memories"):
      ui.label("（空）").classes("text-xs text-gray-400 italic")

  # 更新 summary 层内容列表
  summary_list = refs["summary_list"]
  summary_list.clear()
  with summary_list:
    for m in state.get("summary_memories", []):
      sig = m.get("significance", 0)
      ui.label(
        f"[sig:{sig:.2f}] {m['content']}"
      ).classes("text-xs text-gray-600")
    if not state.get("summary_memories"):
      ui.label("（空）").classes("text-xs text-gray-400 italic")

  # 更新 static 层内容列表
  static_list = refs["static_list"]
  static_list.clear()
  with static_list:
    for m in state.get("static_memories", []):
      category = m.get("category", "")
      prefix = f"[{category}] " if category else ""
      ui.label(
        f"{prefix}{m['content']}"
      ).classes("text-xs text-gray-600")
    if not state.get("static_memories"):
      ui.label("（空）").classes("text-xs text-gray-400 italic")


def _update_topic_card(refs: dict, state: dict) -> None:
  """更新话题管理器卡片"""
  if not refs:
    return

  if state is None:
    refs["status"].text = "话题管理器未启用"
    refs["stats"].text = ""
    refs["timing"].text = ""
    return

  status = "运行中" if state["running"] else "已停止"
  refs["status"].text = f"状态: {status}  模式: {state['comment_mode']}"
  refs["stats"].text = (
    f"话题数: {state['topic_count']}  "
    f"待分类弹幕: {state['pending_comments']}  "
    f"后台任务: {state['background_tasks']}"
  )

  timing = state.get("suggested_timing")
  if timing:
    refs["timing"].text = f"动态等待: {timing[0]:.1f}s ~ {timing[1]:.1f}s"
  else:
    refs["timing"].text = "动态等待: 使用默认值"

  # 更新话题列表
  topic_list = refs["topic_list"]
  topic_list.clear()
  with topic_list:
    topics = state.get("topics", [])
    if not topics:
      ui.label("（无话题）").classes("text-xs text-gray-400 italic")
    for t in topics:
      stale_mark = " [过期]" if t.get("stale") else ""
      title = t.get("title", t["topic_id"])
      with ui.card().classes("w-full p-2").style("background: #f8f9fa"):
        ui.label(
          f"{title} ({t['topic_id']}, sig: {t['significance']:.2f}){stale_mark}"
        ).classes("text-sm font-bold")
        ui.label(f"进度: {t['progress']}").classes("text-xs text-gray-600")
        if t.get("suggestion"):
          ui.label(f"建议: {t['suggestion']}").classes("text-xs text-blue-600")
        ui.label(
          f"弹幕: {t['comment_count']}  用户: {t['user_count']}  "
          f"创建: {t['created_at']}  更新: {t['updated_at']}"
        ).classes("text-xs text-gray-400")


def _update_llm_card(refs: dict, state: dict) -> None:
  """更新 LLM 状态卡片"""
  if not refs or not state:
    return

  refs["model"].text = f"模型: {state['model_type']} ({state['model_name'] or '默认'})"
  refs["persona"].text = f"角色: {state['persona']}"
  refs["history"].text = f"对话历史: {state['history_length']} 轮"
  refs["memory"].text = f"记忆功能: {'已启用' if state['has_memory'] else '未启用'}"
  refs["bg_tasks"].text = f"后台任务: {state['background_tasks']}"


def _update_timing_card(refs: dict, state: dict, studio_state: dict = None) -> None:
  """更新回复耗时分解卡片"""
  if not refs:
    return

  if state is None:
    refs["ts"].text = "（等待首次回复）"
    refs["total"].text = ""
    for field_name, _, _ in _TIMING_STAGES:
      stage_refs = refs["stages"].get(field_name)
      if not stage_refs:
        continue
      stage_refs["bar"].value = 0
      # 未回复时也显示可选阶段的启用状态
      if studio_state and field_name in _STAGE_ENABLED_BY:
        enabled_key, disabled_label = _STAGE_ENABLED_BY[field_name]
        if not studio_state.get(enabled_key, True):
          stage_refs["label"].text = disabled_label
          continue
      stage_refs["label"].text = "—"
    return

  total = state.get("total_ms", 0)
  refs["ts"].text = f"记录时间: {state.get('timestamp', '')}"
  refs["total"].text = f"端到端总耗时: {total:.1f} ms"

  for field_name, _, _ in _TIMING_STAGES:
    stage_refs = refs["stages"].get(field_name)
    if not stage_refs:
      continue
    # 可选阶段：未启用时直接标注，不显示 0 ms
    if studio_state and field_name in _STAGE_ENABLED_BY:
      enabled_key, disabled_label = _STAGE_ENABLED_BY[field_name]
      if not studio_state.get(enabled_key, True):
        stage_refs["bar"].value = 0
        stage_refs["label"].text = disabled_label
        continue
    ms = state.get(field_name, 0)
    ratio = ms / total if total > 0 else 0
    stage_refs["bar"].value = min(ratio, 1.0)
    stage_refs["label"].text = f"{ms:.1f} ms"


def _update_prompt_card(refs: dict, state: dict) -> None:
  """更新 prompt 展示卡片"""
  if not refs or not state:
    return

  # 使用完整 prompt（包含系统提示词）
  prompt = state.get("last_full_prompt") or state.get("last_prompt")
  refs["prompt_text"].content = prompt if prompt else "（等待首次触发）"
