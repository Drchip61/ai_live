# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI Live 是一个 VLM 虚拟直播间系统。AI 主播同时理解直播画面和弹幕，结合结构化记忆系统做出角色化回应。使用 Python + LangChain 构建，全异步架构。

## 开发命令

```bash
# 虚拟环境
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# 主入口：远程模式（截图拉取 + 推送式弹幕 + TTS 广播）
python run_remote.py \
  --screenshot-url "http://10.81.7.165:8000/screenshot" \
  --speech-url "http://10.81.7.165:9200/say" \
  --persona mio \
  --model openai --model-name gpt-5.4 \
  --enable-controller --controller-provider openai \
  --danmaku-host 0.0.0.0 --danmaku-port 9100 \
  --callback-port 9201

# VLM 模式（本地视频 + 弹幕 → 多模态 AI 主播）
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

### run_remote.py 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--screenshot-url` | 截图接口 URL | `http://10.81.7.114:8000/screenshot` |
| `--speech-url` | TTS 服务 URL | 无（不传则纯文本） |
| `--persona` | 主播人设 | `karin` |
| `--model` | 主模型 provider | `openai` |
| `--model-name` | 指定模型名称 | provider 默认大模型 |
| `--enable-controller` | 启用 LLM Controller | 关闭 |
| `--controller-provider` | Controller 模型 provider | `openai` |
| `--danmaku-host/port` | 推送式弹幕监听 | `0.0.0.0:9100` |
| `--callback-port` | TTS 完播回调端口 | 无 |
| `--state-card` | 启用主播状态卡 | 关闭 |
| `--no-memory` | 禁用记忆系统 | 启用 |
| `--ephemeral-memory` | 临时记忆（不持久化） | 持久化 |

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

### 单次 VLM 调用管线

系统最关键的数据流，跨越 `streaming_studio/studio.py` → `langchain_wrapper/wrapper.py` → `langchain_wrapper/pipeline.py`：

```
弹幕缓冲区 + 视频帧
       │
       ├── 弹幕到达 → 记忆预热（后台异步预检索，利用等待空闲）
       │
       ├── Controller 调度 → PromptPlan（路由、风格、检索策略）
       │
       ├── 三阶段 prompt 解析：Controller → Retriever → Composer
       │   记忆/人设/语料检索 → 结构化上下文 → 组装完整 prompt
       │
       ▼
  VLM 调用（大模型 Sonnet/GPT-5.4/Gemini）
  图片直传模型，上下文被 wrap_untrusted_context() 沙箱包装
       │
       ├── 流式路径：_SentenceStreamer 边生成边入队 SpeechQueue
       │                                （首句即播，无需等完整回复）
       ├── 非流式路径：整体解析后入队
       │
       ▼
  记忆写入（小模型，异步后台 task）
  MemoryManager.record_interaction() → 总结为第一人称记忆
```

纯对话模式（无视频/黑屏）不传图片，聚焦弹幕互动。

### SpeechQueue 双循环架构

`StreamingStudio` 在有 TTS 广播时启用 Producer + Dispatcher 双循环：

- **Producer**：收集弹幕 → Controller 调度 → LLM 生成 → 拆句入队，不等 TTS
- **Dispatcher**：从 SpeechQueue 取最高优先级 → 发送 TTS → 等完播 → 触发回调
- **优先级**：`0=付费事件` > `1=弹幕回复` > `2=入场/小礼物` > `3=视频解说/独白`
- **TTL 驱逐**：队满时驱逐优先级最低的条目，过期条目自动跳过

### LLM Controller（集成器架构）

`llm_controller/` — 统一场景化调度，替代旧的 reply_decider：

- **规则路由层** (`rule_router.py`)：处理确定性场景（付费事件、入场、沉默、主动发言），零 LLM 开销
- **并行专家组** (`experts.py`)：规则无法决定时，4 个专家并行调用，各自独立超时/回退
  - `ReplyJudge` — 是否回复 + 紧急度
  - `StyleAdvisor` — 风格 + 句数 + 语气
  - `ContextAdvisor` — 记忆策略 + 话题分类 + 会话锚点
  - `ActionGuard` — 检测不可执行操作请求
- **集成器** (`controller.py`)：合并专家结果 + 规则增强 → 输出 `PromptPlan`

`PromptPlan` 结构化输出包含：路由类型、风格、句数、记忆策略、人设段落、语料检索、观众焦点、额外指令等。

### 双轨制触发机制

`StreamingStudio._main_loop()` 的核心节奏控制：

- **定时器轨**：每轮 `random(min_interval, max_interval)` 秒后触发
- **弹幕加速轨**：每条新弹幕缩短剩余等待时间（`comment_wait_reduction` 秒）
- **话题管理器**可通过 `suggested_timing` 动态覆盖等待区间

弹幕分割逻辑（`_collect_comments`）：以 `_last_collect_time`（开始生成回复的时间点）为分界，区分旧弹幕（背景参考）和新弹幕（互动目标）。回复生成期间到达的弹幕在下一轮仍被视为"新弹幕"。`priority` 弹幕无论时间戳始终归入新弹幕。

### 结构化记忆系统

`memory/` 模块，由 `MemoryManager` 编排，检索由 `StructuredRetriever` 执行：

| 层 | 存储 | 写入时机 | 特点 |
|---|---|---|---|
| Active | 内存 FIFO | 每次交互后（LLM 总结为第一人称记忆） | 满了溢出到 Temporary |
| Temporary | Chroma 向量库 | Active 溢出时 | significance 衰减，低于阈值自动遗忘 |
| Summary | Chroma 向量库 | 定时汇总任务（默认 60s 间隔） | Active + 近期交互 → 小 LLM 汇总 |
| Static | Chroma 向量库 | 启动时从 `personas/{name}/static_memories/*.json` 加载 | 永久记忆（身份、性格） |
| Viewer | JSON 文件 | 交互后异步更新 | 每位观众的关系档案（熟悉度、信任、话题线索） |
| Corpus | JSON 文件 | 启动时从 `data/memory_store/structured/corpus_store.json` 加载 | 风格语料库（按类别/场景标签检索） |

Embedding 模型：`BAAI/bge-small-zh-v1.5`。记忆默认持久化到磁盘（`--ephemeral-memory` 切换为会话级）。

### 话题管理器（可选）

`topic_manager/`，通过 `--topic-manager` 启用：

- **弹幕分类**：单条模式（即时分类）或批量模式（攒够阈值一起分类），规则匹配优先、小模型降级
- **回复后分析**：内容分析（话题进度更新、新话题发现）+ 节奏分析（标记过期话题、动态调整等待时间），两个分析并行执行
- **格式化输出**：为 prompt 添加弹幕→话题标注和话题摘要

### 模型分级策略

`langchain_wrapper/model_provider.py` 中 `REMOTE_MODELS` 定义了大/小模型预设：

- **大模型**（主对话）：GPT-5.4 / Claude Sonnet 4.6 / Gemini 3 Flash
- **小模型**（记忆总结、弹幕分类、Controller 专家）：GPT-5 Mini / Claude Haiku 4.5 / Gemini 2.5 Flash Lite

通过 `ModelProvider.remote_large()` / `remote_small()` 工厂方法获取。`small_model_type` 参数支持大小模型使用不同 provider。

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
- `controller/` — Controller 专家 prompt（reply_judge / style_advisor / context_advisor / action_guard）
- `routes/` — 路由专用 prompt（chat / vlm / proactive / gift / guard_buy / entry / super_chat）
- `studio/` — 直播间专用（弹幕头部、互动指令等）
- `topic/` — 话题管理器专用（分类、分析模板）
- `memory/` — 记忆系统专用（交互总结、定时汇总、观众摘要）
- `state/` — 状态卡系统（初始化、轮次更新）

角色提示词在 `personas/{name}/system_prompt.txt`。完整 system prompt = base_instruction + anti_injection + persona prompt。

### 异步后台任务管理

各模块维护 `_background_tasks: set[asyncio.Task]` 集合，通过 `task.add_done_callback(self._background_tasks.discard)` 自动清理。`stop()` 时统一取消并等待。

## 依赖方向

```
connection → streaming_studio → langchain_wrapper（多模态 images 参数）
                              → llm_controller（集成器调度 → PromptPlan）
                              → memory（结构化检索 + 交互记录）
                              → video_source（帧提取 + 弹幕注入）
                              → topic_manager（弹幕分类 + 上下文格式化）
langchain_wrapper → prompts → personas（system_prompt + static_memories）
debug_console → streaming_studio / langchain_wrapper / memory（debug_state 聚合）
```

## 可用角色

- **karin** — 元气偶像少女
- **sage** — 知性学者
- **kuro** — 王者荣耀技术流主播
- **mio** — 温柔治愈系主播
- **naixiong** — 奶凶（含情绪系统 + 好感度系统）
- **dacongming** — 大聪明

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
