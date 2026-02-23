"""
NiceGUI 应用入口
顶级菜单栏 + 两个可切换子页面（监控面板 / 模拟直播间）
"""

import sys
from pathlib import Path
from typing import Optional

from nicegui import ui

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio

from .state_collector import StateCollector
from .pages.monitor import create_monitor_page
from .pages.chat import create_chat_page


def run(
  model_type: ModelType = ModelType.OPENAI,
  model_name: Optional[str] = None,
  persona: str = "karin",
  port: int = 8080,
  enable_global_memory: bool = False,
  enable_topic_manager: bool = False,
) -> None:
  """
  启动调试控制台

  Args:
    model_type: 模型类型
    model_name: 模型名称
    persona: 角色名称
    port: 监听端口
    enable_global_memory: 是否开启全局记忆（持久化到文件）
    enable_topic_manager: 是否启用话题管理器
  """
  # 初始化直播间
  studio = StreamingStudio(
    persona=persona,
    model_type=model_type,
    model_name=model_name,
    enable_memory=True,
    enable_global_memory=enable_global_memory,
    enable_topic_manager=enable_topic_manager,
  )
  studio.enable_streaming = True
  collector = StateCollector(studio)

  @ui.page("/")
  def index():
    _build_page(studio, collector, default_tab="monitor")

  @ui.page("/monitor")
  def monitor():
    _build_page(studio, collector, default_tab="monitor")

  @ui.page("/chat")
  def chat():
    _build_page(studio, collector, default_tab="chat")

  ui.run(
    title="调试控制台 — mio-streaming-demo",
    port=port,
    reload=False,
  )


def _build_page(
  studio: StreamingStudio,
  collector: StateCollector,
  default_tab: str = "monitor",
) -> None:
  """
  构建带顶部菜单栏的页面框架

  Args:
    studio: 直播间实例
    collector: 状态收集器
    default_tab: 默认激活的标签页
  """
  # 顶部菜单栏
  with ui.header().classes("bg-blue-800 text-white items-center"):
    ui.label("mio-streaming-demo 调试控制台").classes("text-lg font-bold")
    ui.space()
    with ui.tabs().classes("text-white") as tabs:
      monitor_tab = ui.tab("monitor", label="监控面板")
      chat_tab = ui.tab("chat", label="模拟直播间")
    tabs.value = default_tab

  # 页面内容区域
  with ui.tab_panels(tabs).classes("w-full flex-1"):
    with ui.tab_panel("monitor"):
      create_monitor_page(collector)
    with ui.tab_panel("chat"):
      create_chat_page(studio)
