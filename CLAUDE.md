# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI Live 是一个 VLM 虚拟直播间系统。AI 主播同时理解直播画面和弹幕，结合四层记忆系统做出角色化回应。使用 Python + LangChain 构建，全异步架构。

## 开发命令

```bash
# 虚拟环境
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# 主入口：VLM 模式（视频 + 弹幕 → 多模态 AI 主播）
python run_vlm_demo.py --video data/sample.mp4 --danmaku data/sample.xml
python run_vlm_demo.py --video data/sample.mp4 --persona kuro --model anthropic --speed 4.0

# NiceGUI 调试控制台（监控面板 + 模拟直播间）
python -m debug_console
python -m debug_console --port 8080 --persona karin --model openai

# 弹幕模拟测试（随机用户身份，不需要视频文件）
python -m streaming_studio.test_danmaku_studio

# B站视频下载（需 pip install yt-dlp，生成 .mp4 + .xml 到 data/）
python download_bilibili.py "https://b23.tv/xxxxx"
```

### API Key 配置

创建 `secrets/api_keys.json`（已 gitignore）或设置环境变量：

```json
{
  "anthropic_api_key": "sk-ant-...",
  "openai_api_key": "sk-...",
  "gemini_api_key": "..."
}
```

环境变量名：`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`（优先级高于文件）。

## 核心架构

### 两趟 VLM 调用管线

这是系统最关键的数据流，跨越 `streaming_studio/studio.py` → `langchain_wrapper/wrapper.py` → `langchain_wrapper/pipeline.py`：

```
弹幕缓冲区 + 视频帧
       │
       ▼
  第一趟：场景理解 (小模型 Haiku/Mini)
  wrapper.ascene_understand() → 客观描述画面+弹幕
  不使用人设/记忆/历史，纯视觉+文本理解
       │
       ▼
  记忆检索 (RAG)
  用场景描述作为 query 搜索四层记忆 → extra_context
       │
       ▼
  第二趟：完整回复 (大模型 Sonnet/GPT)
  wrapper.achat() → system_prompt + extra_context + 画面 + 弹幕
  extra_context 被 wrap_untrusted_context() 包装为只读参考数据
       │
       ▼
  记忆写入 (小模型，异步后台 task)
  MemoryManager.record_interaction() → 总结为第一人称记忆
```

纯文本模式（无视频）跳过第一趟，直接用弹幕内容做 RAG 查询。

### 双轨制触发机制

`StreamingStudio._main_loop()` 的核心节奏控制：

- **定时器轨**：每轮 `random(min_interval, max_interval)` 秒后触发
- **弹幕加速轨**：每条新弹幕缩短剩余等待时间（`comment_wait_reduction` 秒）
- **话题管理器**可通过 `suggested_timing` 动态覆盖等待区间

弹幕分割逻辑（`_collect_comments`）：以 `_last_collect_time`（开始生成回复的时间点）为分界，区分旧弹幕（背景参考）和新弹幕（互动目标）。回复生成期间到达的弹幕在下一轮仍被视为"新弹幕"。`priority` 弹幕无论时间戳始终归入新弹幕。

### 回复决策器（两阶段）

`streaming_studio/reply_decider.py`：

- **Phase 1 规则快筛**（免费）：必须回复（提问、高活跃、优先弹幕）或建议跳过（纯反应词、刷屏）
- **Phase 2 LLM 精判**（小模型）：规则无法决定时，综合弹幕内容和沉默时长判断，返回 `{reply, urgency, reason}` JSON
- **主动发言**：沉默超阈值 + 画面变化时触发 `should_proactive_speak()`

### 四层记忆系统

`memory/` 模块，由 `MemoryManager` 编排：

| 层 | 存储 | 写入时机 | 特点 |
|---|---|---|---|
| Active | 内存 FIFO | 每次交互后（LLM 总结为第一人称记忆） | 满了溢出到 Temporary |
| Temporary | Chroma 向量库 | Active 溢出时 | significance 衰减，低于阈值自动遗忘 |
| Summary | Chroma 向量库 | 定时汇总任务（默认 60s 间隔） | Active + 近期交互 → 小 LLM 汇总 |
| Static | Chroma 向量库 | 启动时从 `personas/{name}/static_memories/*.json` 加载 | 永久记忆（身份、性格） |

检索由 `MemoryRetriever` 跨层执行，支持 quota（每层定额）和 weighted（加权合并）两种模式。Embedding 模型：`BAAI/bge-small-zh-v1.5`。

全局记忆（`--global-memory`）持久化到磁盘；默认模式使用 Chroma EphemeralClient，会话结束即丢弃。

### 话题管理器（可选）

`topic_manager/`，通过 `--topic-manager` 启用：

- **弹幕分类**：单条模式（即时分类）或批量模式（攒够阈值一起分类），规则匹配优先、小模型降级
- **回复后分析**：内容分析（话题进度更新、新话题发现）+ 节奏分析（标记过期话题、动态调整等待时间），两个分析并行执行
- **格式化输出**：为 prompt 添加弹幕→话题标注和话题摘要

### 模型分级策略

`langchain_wrapper/model_provider.py` 中 `REMOTE_MODELS` 定义了大/小模型预设：

- **大模型**（主对话）：Sonnet / GPT-5.2 / Gemini 2.5 Flash
- **小模型**（场景理解、记忆总结、弹幕分类、回复决策）：Haiku / GPT-5 Mini / Gemini 2.0 Flash

通过 `ModelProvider.remote_large()` / `remote_small()` 工厂方法获取。Gemini 走 OpenAI 兼容接口。

## 关键设计模式

### 全链路抗注入

三层防护，分布在多个文件中：

1. **弹幕清洗** (`studio._sanitize_comment_for_prompt`)：正则匹配注入特征，命中时添加"不可执行"标记
2. **用户输入护栏** (`wrapper._guard_user_input`)：疑似注入时包装为 `[BEGIN_USER_INPUT]...[END_USER_INPUT]`
3. **动态上下文沙箱** (`pipeline.wrap_untrusted_context`)：记忆/话题等 extra_context 包装为 `[BEGIN_UNTRUSTED_CONTEXT]...[END_UNTRUSTED_CONTEXT]`，声明为不可信参考数据

### debug_state() 模式

几乎所有核心类（`StreamingStudio`、`LLMWrapper`、`MemoryManager`、`TopicManager`、`VideoPlayer`）都暴露 `debug_state() -> dict` 方法。`debug_console/state_collector.py` 聚合这些快照供 NiceGUI 监控面板实时显示。

### 提示词文件结构

所有提示词以 `.txt` 文件存放在 `prompts/` 下，通过 `PromptLoader` 加载：

- `base_instruction.txt` — 主播基础指令
- `security/anti_injection.txt` — 安全规则
- `studio/` — 直播间专用（场景理解、弹幕头部、回复决策等）
- `topic/` — 话题管理器专用（分类、分析模板）
- `memory/` — 记忆系统专用（交互总结、定时汇总）

角色提示词在 `personas/{name}/system_prompt.txt`。完整 system prompt = base_instruction + anti_injection + persona prompt。

### 异步后台任务管理

各模块维护 `_background_tasks: set[asyncio.Task]` 集合，通过 `task.add_done_callback(self._background_tasks.discard)` 自动清理。`stop()` 时统一取消并等待。

## 依赖方向

```
connection → streaming_studio → langchain_wrapper（多模态 images 参数）
                              → memory（RAG 检索 + 交互记录）
                              → video_source（帧提取 + 弹幕注入）
                              → topic_manager（弹幕分类 + 上下文格式化）
langchain_wrapper → prompts → personas（system_prompt + static_memories）
debug_console → streaming_studio / langchain_wrapper / memory（debug_state 聚合）
```

## 可用角色

- **karin** — 元气偶像少女
- **sage** — 知性学者
- **kuro** — 王者荣耀技术流主播

添加新角色：在 `personas/` 下创建子目录，放入 `system_prompt.txt` 和 `static_memories/` 目录。

## 编码规范

- 命名：snake_case
- 缩进：2 空格
- 注释/文档：中文
- 路径以项目根目录为基准
- 数据类使用 `@dataclass(frozen=True)`
- `spec/` 目录为只读规范文档，不要修改

## 当前阶段

草稿阶段：搭建框架，代码简洁可读，不考虑生产环境部署需求。
