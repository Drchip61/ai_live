# AI Live — VLM 虚拟主播系统

基于视觉语言模型的虚拟直播间系统。AI 主播同时理解直播画面和弹幕内容，经由五层管线做出角色化回应，支持句级流式 TTS、结构化记忆和主播状态卡。

## 五层管线架构

```
弹幕 + 画面帧
      │
      ▼
┌──────────────────────────────────────────────────┐
│ ① Controller — 场景调度与决策                      │
│   规则路由（付费/入场/沉默）→ 并行专家组降级         │
│   输出 PromptPlan（路由、风格、检索策略、副作用）     │
└───────────────────────┬──────────────────────────┘
                        │ PromptPlan
                        ▼
┌──────────────────────────────────────────────────┐
│ ② Retriever — 上下文检索                          │
│   按检索计划从 8 类记忆索引异步检索                  │
│   输出 RetrievedContextBundle（trusted/untrusted） │
└───────────────────────┬──────────────────────────┘
                        │ ContextBundle
                        ▼
┌──────────────────────────────────────────────────┐
│ ③ Composer — Prompt 组装                          │
│   路由模板 + 风格提示 + 检索上下文 → 模型调用载荷    │
│   untrusted 上下文沙箱包装，trusted 注入 system 侧  │
└───────────────────────┬──────────────────────────┘
                        │ ModelInvocation
                        ▼
┌──────────────────────────────────────────────────┐
│ ④ Replyer — LLM 生成与播放                        │
│   大模型调用（GPT-5.4 / Sonnet 4.6 / Gemini 3）   │
│   _SentenceStreamer 边生成边入队，首句即播          │
└───────────────────────┬──────────────────────────┘
                        │ StreamerResponse → SpeechQueue
                        ▼
┌──────────────────────────────────────────────────┐
│ ⑤ Updater — 记忆写入与状态更新（异步后台）          │
│   三路并行：交互总结 / 观众档案提取 / 自我立场提取   │
│   StateCard 规则衰减 + LLM 叙事更新               │
└──────────────────────────────────────────────────┘
```

### 各层概览

| 层 | 关键文件 | 输入 → 输出 |
|---|---|---|
| **Controller** | `llm_controller/controller.py` `rule_router.py` `experts.py` | `ControllerInput` → `PromptPlan` |
| **Retriever** | `langchain_wrapper/retriever.py` `memory/structured_retriever.py` | `PromptPlan` → `RetrievedContextBundle` |
| **Composer** | `streaming_studio/route_composer.py` | `PromptPlan` + `ContextBundle` + 弹幕/画面 → `ModelInvocation` |
| **Replyer** | `streaming_studio/studio.py` `langchain_wrapper/wrapper.py` | `ModelInvocation` → `StreamerResponse` → `SpeechQueue` |
| **Updater** | `memory/manager.py` `broadcaster_state/state_updater.py` | 回复 + 弹幕 → 记忆写入 + StateCard 更新 |

### 调用链路

```
_main_loop
  → _collect_comments()
  → _main_loop_controller_round()                           ← Controller
    → LLMController.dispatch()
    → _generate_and_enqueue_with_plan()
      → _resolve_prompt_invocation_with_plan()
        → RetrieverResolver.resolve()                        ← Retriever
        → PromptComposer.compose()                           ← Composer
      → LLMWrapper.achat_stream_with_plan()                  ← Replyer
        → _SentenceStreamer → SpeechQueue 入队
        → _schedule_memory_writeback()                       ← Updater
      → _schedule_state_round_update()                       ← Updater
```

## 子系统说明

### LLM Controller（集成器架构）

- **规则路由层**：处理确定性场景（付费事件、入场、沉默、主动发言），零 LLM 开销，同时产出 `RuleEnrichment`（人设段落、知识命中、关系牌）
- **并行专家组**：规则无法决定时，4 个小模型专家并行调用
  - `ReplyJudge` — 是否回复 + 紧急度
  - `StyleAdvisor` — 风格 + 句数 + 语气
  - `ContextAdvisor` — 记忆策略 + 话题分类 + 会话锚点
  - `ActionGuard` — 检测不可执行操作请求
- **集成器**：合并规则信号 + 专家结果 → `PromptPlan`（含 `ReplyDecision` / `RetrievalPlan` / `SideEffectPlan`）

### 结构化记忆系统

| 层 | 存储 | 写入时机 | 特点 |
|---|---|---|---|
| Active | 内存 FIFO | 每次交互后 | 满了溢出到 Temporary |
| Temporary | Chroma 向量库 | Active 溢出时 | significance 衰减，自动遗忘 |
| Summary | Chroma 向量库 | 定时汇总（60s） | Active + 近期交互 → 小 LLM 汇总 |
| Static | Chroma 向量库 | 启动时加载 | 预设永久记忆（身份、性格） |
| Viewer | JSON 文件 | 交互后异步更新 | 每位观众的结构化关系档案 |
| Corpus | JSON 文件 | 启动时加载 | 风格语料库（按类别/场景检索） |

`StructuredMemoryRetriever` 维护 8 个向量投影索引，覆盖用户事实/回钩、自我立场/承诺/线头、人设、语料、外部知识。弹幕到达时触发**记忆预热**（后台异步预检索），利用等待空闲减少响应延迟。

### SpeechQueue 双循环架构

- **Producer 循环**：收集弹幕 → Controller → Replyer → 拆句入队，不等 TTS
- **Dispatcher 循环**：从队列取最高优先级条目 → 发送 TTS → 等完播回调
- 优先级：`0=付费事件` > `1=弹幕回复` > `2=入场/小礼物` > `3=视频解说/独白`
- TTL 驱逐：队满时驱逐最低优先级，过期条目自动跳过

### 双轨制触发机制

- **定时器轨**：每轮随机等待 `min_interval ~ max_interval` 秒
- **弹幕加速轨**：每条新弹幕缩短剩余等待时间
- 话题管理器可通过 `suggested_timing` 动态覆盖等待区间

## 快速开始

### 安装依赖

```bash
# Python 3.9+
pip install -r requirements.txt

# 下载 B站视频（可选）
pip install yt-dlp
```

### 配置 API Key

创建 `secrets/api_keys.json`：

```json
{
  "anthropic_api_key": "sk-ant-...",
  "openai_api_key": "sk-...",
  "gemini_api_key": "..."
}
```

或通过环境变量：`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY`（优先级高于文件）。

### 运行（远程模式 — 主要入口）

```bash
python run_remote.py \
  --screenshot-url "http://10.81.7.165:8000/screenshot" \
  --speech-url "http://10.81.7.165:9200/say" \
  --persona mio \
  --model openai --model-name gpt-5.4 \
  --enable-controller --controller-provider openai \
  --danmaku-host 0.0.0.0 --danmaku-port 9100 \
  --callback-port 9201
```

### 常用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--screenshot-url` | 截图接口 URL | `http://10.81.7.114:8000/screenshot` |
| `--speech-url` | TTS 服务 URL | 无（纯文本模式） |
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

### 其他运行方式

```bash
# VLM 模式（本地视频 + 弹幕）
python run_vlm_demo.py --video data/sample.mp4 --danmaku data/sample.xml --persona kuro --model anthropic

# NiceGUI 调试控制台
python -m debug_console --port 8080 --persona karin --model openai

# 弹幕模拟测试
python -m streaming_studio.test_danmaku_studio

# B站视频下载
python download_bilibili.py "https://b23.tv/xxxxx"
```

## 模型分级策略

| 级别 | 用途 | OpenAI | Anthropic | Gemini |
|------|------|--------|-----------|--------|
| 大模型 | 主对话 | GPT-5.4 | Claude Sonnet 4.6 | Gemini 3 Flash |
| 小模型 | 记忆/分类/Controller | GPT-5 Mini | Claude Haiku 4.5 | Gemini 2.5 Flash Lite |

另支持 DeepSeek（`deepseek-chat`）和 Qwen（`qwen3.5-plus` / `qwen3.5-flash`）。`small_model_type` 参数允许大小模型使用不同 provider。

## 项目结构

```
streaming_studio/       # 五层管线编排、SpeechQueue、双轨定时器
llm_controller/         # Controller 层（规则路由 + 并行专家 + 集成器）
langchain_wrapper/      # Retriever + Replyer 层（检索调度、LLM 调用管线）
memory/                 # 结构化记忆系统（6 层存储 + 向量检索引擎）
broadcaster_state/      # Updater 层（StateCard 更新）
personas/               # 角色人设（system prompt + 预设记忆）
prompts/                # 提示词模板（路由/Controller/记忆/状态卡）
style_bank/             # 风格语料库
topic_manager/          # 话题管理器（弹幕分类、节奏分析）
connection/             # 远程数据源（截图拉取、弹幕推送、TTS 广播）
debug_console/          # NiceGUI 调试控制台
video_source/           # 本地视频源（帧提取、弹幕解析）
secrets/                # API 密钥（已 gitignore）
data/                   # 视频/弹幕/记忆存储（大文件已 gitignore）
```

## 可用角色

| 角色 | 说明 |
|------|------|
| **karin** | 元气偶像少女 |
| **sage** | 知性学者 |
| **kuro** | 王者荣耀技术流主播 |
| **mio** | 温柔治愈系主播 |
| **naixiong** | 奶凶（情绪系统 + 好感度系统） |
| **dacongming** | 大聪明 |

添加新角色：在 `personas/` 下创建子目录，放入 `system_prompt.txt` 和 `static_memories/` 目录。
