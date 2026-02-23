# debug_console — NiceGUI 调试控制台

## 概述

基于 NiceGUI 的本地 Web 调试控制台，提供两个可切换的子页面：
1. **监控面板** — 实时显示后台各模块的运行状态
2. **模拟直播间** — 单人测试环境，支持固定用户和随机用户模式

独立端口运行（默认 8080），不影响现有 connection 层的 WebSocket 服务。

## 模块结构

```
debug_console/          # 新顶层模块
  __init__.py           #   导出 DebugConsole
  app.py                #   NiceGUI 应用入口 + 顶级菜单路由
  pages/
    __init__.py
    monitor.py          #   监控面板页面
    chat.py             #   模拟直播间页面
  state_collector.py    #   状态收集器（从各模块读取 debug_state）
```

## 对现有代码的改动

仅新增 `debug_state()` 方法，不改动任何现有逻辑：

| 文件 | 改动 |
|------|------|
| `streaming_studio/studio.py` | 新增 `debug_state() -> dict` |
| `langchain_wrapper/wrapper.py` | 新增 `debug_state() -> dict` |
| `memory/manager.py` | 新增 `debug_state() -> dict` |

每个 `debug_state()` 只读取已有属性，零副作用。

## 页面设计

### 页面 1：监控面板 (`/monitor`)

全部垂直排列（无左右并列），顺序为：

#### 1. 直播间状态卡片
- 运行状态、弹幕缓冲区大小 / 最大容量
- 定时器参数（min_interval / max_interval）
- 上次回复时间、待处理弹幕数
- 最近弹幕列表（时间 + 用户 + 内容）

#### 2. LLM 状态卡片
- 模型类型、模型名称、角色
- 对话历史长度
- 后台任务数量

#### 3. 最近 Prompt 卡片
- 最近一次发给模型的完整 prompt（系统提示词 + 记忆上下文 + 当前消息）

#### 4. 记忆系统卡片
- active 层容量进度条 + 当前内容列表
- temporary / summary / static 层：可展开面板，显示各层全部内容
- 近期交互缓冲数量
- 定时任务状态（汇总任务运行中/清理任务运行中）

#### 刷新机制
- `ui.timer(2.0, refresh)` 每 2 秒自动刷新
- 手动刷新按钮

### 页面 2：模拟直播间 (`/chat`)

#### 左右分栏布局
- **左栏：主播发言** — 蓝色背景滚动区域，独立滚动
  - 非流式：完整回复一次性显示
  - 流式：逐 token 更新气泡文字（`ui.label.set_text()`），完成后添加时间戳
  - 使用 `streamed_ids`（有界 deque）去重，避免 `on_response` 重复创建气泡
- **右栏：观众弹幕 + 输入** — 灰色背景滚动区域，独立滚动
  - 弹幕气泡（显示昵称 + 时间 + 内容）
  - 输入框 + 发送按钮

#### 用户模式
- **单用户模式** — 固定 user_id 和昵称（手动设置）
- **多用户模式** — 每条弹幕自动生成随机身份（coolname）
- 当前身份预览（多用户模式下显示下一个将使用的随机身份）

#### 控制区域
- 启动/停止直播间
- 运行状态显示

### 顶级菜单栏

使用 `ui.header` + `ui.tabs` 实现页面切换：
```
[监控面板] [模拟直播间]
```

## 数据流

```
debug_console/app.py
  │
  ├── 持有 StreamingStudio 实例引用
  │
  ├── /monitor 页面
  │     └── ui.timer(2.0) → state_collector.snapshot()
  │           ├── studio.debug_state()
  │           ├── studio.llm_wrapper.debug_state()
  │           └── studio.llm_wrapper._memory.debug_state()
  │
  └── /chat 页面
        ├── 发送弹幕 → studio.send_comment(Comment(...))
        └── studio.on_response(callback) → 显示回复气泡
```

## debug_state() 返回值设计

### StreamingStudio.debug_state()
```python
{
  "is_running": bool,
  "min_interval": float,
  "max_interval": float,
  "buffer_size": int,
  "buffer_max": int,
  "pending_comment_count": int,
  "last_reply_time": Optional[str],    # ISO格式
  "last_prompt": Optional[str],        # 最近一次格式化的完整 prompt
  "recent_comments": [                 # 最近 N 条弹幕
    {"nickname": str, "content": str, "timestamp": str}, ...
  ],
  "total_comments": int,
  "total_responses": int,
}
```

### LLMWrapper.debug_state()
```python
{
  "model_type": str,
  "model_name": Optional[str],
  "persona": str,
  "history_length": int,
  "has_memory": bool,
  "background_tasks": int,
  "system_prompt_preview": str,        # 前 200 字符
}
```

### MemoryManager.debug_state()
```python
{
  "active_count": int,
  "active_capacity": int,
  "active_memories": [{"content": str, "timestamp": str}, ...],
  "temporary_count": int,
  "temporary_memories": [{"content": str, "timestamp": str, "significance": float}, ...],
  "summary_count": int,
  "summary_memories": [{"content": str, "timestamp": str, "significance": float}, ...],
  "static_count": int,
  "static_memories": [{"content": str, "category": str}, ...],
  "recent_interactions": int,
  "summary_task_running": bool,
  "cleanup_task_running": bool,
}
```

## 实现顺序

1. 各模块添加 `debug_state()` 方法
2. `state_collector.py` — 聚合各模块状态
3. `app.py` — NiceGUI 应用骨架 + 菜单路由
4. `pages/monitor.py` — 监控面板
5. `pages/chat.py` — 模拟直播间
6. 更新 CLAUDE.md 和 plan/README.md

## 启动方式

```bash
python -m debug_console
```
或：
```bash
python -m debug_console --port 8080 --persona karin --model openai
```
