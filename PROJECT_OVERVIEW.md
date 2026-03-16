# AI Live — VLM 虚拟直播间系统

> AI 主播同时理解直播画面和弹幕，结合六层记忆系统做出角色化回应。Python + LangChain 构建，全异步架构。

---

## 目录

- [系统总览](#系统总览)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心架构](#核心架构)
  - [单次 VLM 调用管线](#单次-vlm-调用管线)
  - [双轨制触发机制](#双轨制触发机制)
  - [弹幕收集与分割](#弹幕收集与分割)
  - [三种运行模式](#三种运行模式)
  - [弱智吧风格注入（StyleBank）](#弱智吧风格注入stylesheet)
- [模块详解](#模块详解)
  - [streaming_studio — 虚拟直播间核心](#streaming_studio--虚拟直播间核心)
  - [langchain_wrapper — LLM 交互层](#langchain_wrapper--llm-交互层)
  - [memory — 六层记忆系统](#memory--六层记忆系统)
  - [video_source — 视频源](#video_source--视频源)
  - [topic_manager — 话题管理器](#topic_manager--话题管理器)
  - [connection — 外部连接层](#connection--外部连接层)
  - [debug_console — 调试控制台](#debug_console--调试控制台)
  - [prompts — 提示词模板](#prompts--提示词模板)
  - [personas — 角色人设](#personas--角色人设)
  - [emotion — 情绪系统](#emotion--情绪系统)
  - [meme — 梗管理](#meme--梗管理)
  - [style_bank — 风格参考库](#style_bank--风格参考库)
  - [expression_mapper — 表情动作映射](#expression_mapper--表情动作映射)
  - [validation — 回复校验](#validation--回复校验)
- [数据流总图](#数据流总图)
- [模型调用全景图](#模型调用全景图)
- [全链路抗注入](#全链路抗注入)
- [配置体系](#配置体系)
- [入口点汇总](#入口点汇总)
- [角色一览](#角色一览)
- [依赖清单](#依赖清单)

---

## 系统总览

AI Live 是一个 VLM（Vision-Language Model）虚拟主播直播系统。系统接收视频流（或远程截图）和弹幕输入，通过多模态大模型生成角色化的主播回复，同时维护长期记忆、追踪话题流动、管理情绪状态。

**核心能力：**

| 能力 | 说明 |
|---|---|
| 多模态理解 | 同时理解直播画面（视频帧）和弹幕文本 |
| 角色化回应 | 5 个预设角色，各具独特性格和记忆 |
| 六层记忆 | Active → Temporary → Summary → Static → Stance → Viewer |
| 话题追踪 | 弹幕分类、话题进度管理、动态节奏调整 |
| 情绪系统 | 5 种情绪状态 + 好感度档位（奶凶角色专用） |
| 梗生命周期 | 自动发现梗、追踪热度、成熟后注入对话 |
| 抗注入防护 | 三层防线：弹幕清洗 → 用户输入护栏 → 动态上下文沙箱 |
| 流式输出 | 逐 token 推送，支持 TTS 完播同步 |
| 跨会话持久 | 记忆默认持久化到磁盘，观众记忆跨天保留 |

---

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
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

### 运行

```bash
# VLM 模式（视频 + 弹幕 → 多模态 AI 主播）
python run_vlm_demo.py --video data/sample.mp4 --danmaku data/sample.xml
python run_vlm_demo.py --video data/sample.mp4 --persona kuro --model anthropic --speed 4.0

# NiceGUI 调试控制台（监控面板 + 模拟直播间）
python -m debug_console
python -m debug_console --port 8080 --persona karin --model openai

# 弹幕模拟测试（随机用户身份，不需要视频文件）
python -m streaming_studio.test_danmaku_studio

# B站视频下载
python download_bilibili.py "https://b23.tv/xxxxx"
```

---

## 项目结构

```
ai_live/
│
├── run_vlm_demo.py               # CLI VLM 直播间入口
├── run_vlm_gui.py                # NiceGUI Web GUI 入口（视频画面 + 弹幕侧栏 + 控制台）
├── run_server.py                 # WebSocket 服务器入口
├── run_remote.py                 # 远程数据源模式入口（Pull 截图+弹幕 → VLM → TTS）
├── download_bilibili.py          # B站视频+弹幕下载工具
├── expression_motion_mapping.json # 表情(12种)/动作(10种)映射配置
├── requirements.txt              # Python 依赖
│
├── streaming_studio/             # 虚拟直播间核心引擎
│   ├── studio.py                 #   主循环、弹幕管理、回复生成（~1860行）
│   ├── models.py                 #   数据模型：Comment / StreamerResponse / ResponseChunk
│   ├── config.py                 #   StudioConfig / ReplyDeciderConfig / CommentClustererConfig
│   ├── reply_decider.py          #   两阶段回复决策器 + 弹幕聚类器
│   ├── timer.py                  #   管线阶段耗时统计
│   ├── guard_roster.py           #   舰长/提督/总督名册管理
│   ├── database.py               #   SQLite 弹幕/回复记录
│   ├── test_danmaku_studio.py    #   弹幕模拟测试
│   ├── test_chatter_studio.py    #   CLI 聊天测试
│   └── test_comment_window.py    #   弹幕窗口回归测试
│
├── langchain_wrapper/            # LLM 交互层
│   ├── wrapper.py                #   LLMWrapper：聊天入口 + 注入检测 + 记忆编排（~612行）
│   ├── pipeline.py               #   LCEL 流式管道：消息组装 + 多模态 + 安全沙箱（~504行）
│   └── model_provider.py         #   模型工厂：大/小模型分级，4家供应商
│
├── memory/                       # 六层记忆系统
│   ├── manager.py                #   顶层编排器：初始化、读写、定时汇总（~697行）
│   ├── retriever.py              #   跨层检索器：quota/weighted 模式（~344行）
│   ├── config.py                 #   全局配置（8个子配置 dataclass）
│   ├── store.py                  #   Chroma 向量存储封装
│   ├── archive.py                #   已删除记忆归档到 JSON
│   ├── significance.py           #   significance 评分：衰减/提升/初始值
│   ├── formatter.py              #   记忆格式化为 prompt 文本
│   ├── prompts.py                #   加载记忆相关提示词模板
│   └── layers/                   #   各记忆层实现
│       ├── base.py               #     MemoryEntry 统一数据结构
│       ├── active.py             #     Active 层：内存 FIFO
│       ├── temporary.py          #     Temporary 层：Chroma + significance 衰减
│       ├── summary.py            #     Summary 层：定时汇总 + 定期清理
│       ├── static.py             #     Static 层：从 JSON 预设加载
│       ├── stance.py             #     Stance 层：AI 立场/观点 + 冲突检测
│       ├── viewer.py             #     Viewer 层：按 user_id 索引的观众记忆
│       ├── user_profile.py       #     UserProfile 层：结构化用户画像
│       └── character_profile.py  #     CharacterProfile 层：角色设定档
│
├── video_source/                 # 视频源模块
│   ├── video_player.py           #   异步视频时间轴播放器（双帧率设计）
│   ├── frame_extractor.py        #   OpenCV 帧提取 → base64 JPEG
│   └── danmaku_parser.py         #   B站弹幕 XML 解析器
│
├── topic_manager/                # 话题管理器
│   ├── manager.py                #   话题编排器：分类/分析/种子/独白（~612行）
│   ├── models.py                 #   Topic / ContentAnalysisDelta / RhythmAnalysisDelta
│   ├── config.py                 #   完整配置（分类/话题表/动态等待/独白/种子库）
│   ├── classifier.py             #   两层弹幕分类：规则匹配 → 小模型降级
│   ├── analyzer.py               #   回复后并行分析：内容分析 + 节奏分析
│   ├── formatter.py              #   话题上下文格式化（标注/摘要/跟进建议）
│   ├── table.py                  #   内存话题表：不可变 Topic CRUD
│   └── prompts.py                #   加载话题相关提示词模板
│
├── connection/                   # 外部连接层
│   ├── stream_service_host.py    #   WebSocket 服务主机
│   ├── speech_broadcaster.py     #   语音/动作广播器（TTS + 完播同步）
│   ├── remote_source.py          #   远程数据源（Pull 模式，替代 VideoPlayer）
│   └── test_chatter_web.py       #   WebSocket 测试客户端
│
├── debug_console/                # NiceGUI 调试控制台
│   ├── __main__.py               #   CLI 入口（python -m debug_console）
│   ├── app.py                    #   NiceGUI 应用主框架
│   ├── state_collector.py        #   状态聚合器（debug_state 快照）
│   ├── comment_broadcaster.py    #   跨客户端弹幕/回复广播
│   ├── auto_viewer.py            #   自动观众（小模型生成虚拟弹幕）
│   └── pages/
│       ├── chat.py               #     模拟直播间页面
│       └── monitor.py            #     实时监控面板
│
├── prompts/                      # 提示词模板（28个 txt + 1个 JSON）
│   ├── prompt_loader.py          #   PromptLoader：加载/组装系统提示词
│   ├── base_instruction.txt      #   主播基础指令
│   ├── auto_viewer.txt           #   自动观众弹幕生成
│   ├── security/
│   │   └── anti_injection.txt    #   安全与抗注入规则
│   ├── studio/                   #   直播间专用提示词（互动指令/回复决策/场景理解等）
│   ├── memory/                   #   记忆系统提示词（交互总结/定时汇总/立场提取/观众改写）
│   └── topic/                    #   话题管理器提示词（分类/分析/跟进/独白等）
│
├── personas/                     # 角色人设（5个角色）
│   ├── persona_loader.py         #   PersonaLoader：自动发现角色、加载 system_prompt
│   ├── karin/                    #   元气偶像少女
│   ├── sage/                     #   知性学者
│   ├── kuro/                     #   王者荣耀技术流主播
│   ├── naixiong/                 #   傲娇毒舌二次元伴侣
│   └── dacongming/              #   民间逻辑学家
│
├── emotion/                      # 情绪系统（奶凶角色专用）
│   ├── state.py                  #   EmotionMachine：5种情绪状态机
│   ├── detector.py               #   EmotionTriggerDetector：触发检测
│   └── affection.py              #   AffectionBank：好感度三档系统
│
├── meme/                         # 梗管理
│   ├── manager.py                #   MemeManager：生命周期(growing→mature→retired)
│   └── detector.py               #   MemeDetector：梗信号识别
│
├── style_bank/                   # 风格参考库
│   └── bank.py                   #   StyleBank：语料→Chroma，按情境语义检索
│
├── expression_mapper/            # 表情动作映射
│   └── mapper.py                 #   sentence embedding + cosine similarity 映射
│
├── validation/                   # 回复校验
│   └── checker.py                #   ResponseChecker：角色约束校验
│
├── tools/                        # 辅助工具
│   └── build_style_bank.py       #   风格语料处理管线
│
├── data/                         # 数据目录
│   ├── guard_roster.json         #   舰长名册
│   ├── memory_store/             #   Chroma 向量库持久化
│   └── raw/                      #   原始语料
│
├── plan/                         # 设计文档
└── spec/                         # 只读规范文档
```

---

## 核心架构

### 单次 VLM 调用管线

这是系统最关键的数据流，跨越 `streaming_studio/studio.py` → `langchain_wrapper/wrapper.py` → `langchain_wrapper/pipeline.py`：

```
弹幕缓冲区 + 视频帧
 │
 ├── 弹幕 → 话题管理器分类标注
 │          → 互动目标加权选择
 │          → 格式化为 prompt（含时间戳/徽章/聚类折叠/话题标注）
 │
 ├── 弹幕文本 → RAG 记忆检索
 │   用有意义弹幕（过滤噪音后）作为 query 搜索六层记忆
 │   返回 (active_text, rag_text, viewer_text)
 │
 ├── extra_context 组装（_build_extra_context）：
 │   ① 情绪状态 + 好感度档位
 │   ② 角色设定档（全量）
 │   ③ 用户画像（全量）
 │   ④ 检索记忆（active + RAG + viewer）
 │   ⑤ 活跃梗注入
 │   ⑥ 话题上下文
 │   ⑦ 风格参考库（随机概率触发）
 │
 ▼
 单次 VLM 调用：完整回复
 pipeline.ainvoke():
   SystemMessage = system_prompt + wrap_untrusted_context(extra_context)
   + 历史消息对（最近 max_history 轮）
   + HumanMessage = [图片 content blocks] + [格式化弹幕文本]
   → 大模型（Sonnet / GPT-5.2 / Gemini 3 Flash）
   → 后处理器链
 │
 ├── 表情动作映射（sentence embedding → 固定标签集）
 │
 ▼
 异步后台任务（3 个并行，不阻塞下一轮）：
 ├── record_interaction → 小 LLM 总结为第一人称记忆 → Active 层
 ├── extract_stances → 正则预筛 + 小 LLM 提取立场 → Stance 层
 └── record_viewer_memories → 正则预过滤 + 小 LLM 筛选改写 → Viewer 层
```

纯对话模式（无视频/黑屏）不传图片，聚焦弹幕互动。

### 双轨制触发机制

`StreamingStudio._main_loop()` 的核心节奏控制：

```
┌─ 有 TTS 模式 ──────────────────────────────────────┐
│  await speech_gate() → TTS 完播后进入               │
│  弹幕积累足够 → 跳过额外等待                        │
│  否则短暂等待新弹幕（0.2s 或 2s）                   │
└────────────────────────────────────────────────────┘

┌─ 无 TTS 模式 ──────────────────────────────────────┐
│  remaining = random(min_interval, max_interval)     │
│  或来自 topic_manager.suggested_timing              │
│  或 _proactive_shortcut（空轮快捷路径）             │
│                                                     │
│  while remaining > 0:                               │
│    每条新弹幕缩减 remaining（comment_wait_reduction）│
│    首条弹幕唤醒（_was_silent）→ 立即跳出            │
│    超时 → 跳出                                      │
└────────────────────────────────────────────────────┘
```

- **定时器轨**：每轮 `random(min_interval, max_interval)` 秒后触发
- **弹幕加速轨**：每条新弹幕缩短剩余等待时间（`comment_wait_reduction` 秒）
- **话题管理器**可通过 `suggested_timing` 动态覆盖等待区间

### 弹幕收集与分割

以 `_last_collect_time`（开始生成回复的时间点）为分界，区分旧弹幕（背景参考）和新弹幕（互动目标）。

```
时间线: ──────[_last_collect_time]──────────[now]──────
                    │                        │
        ┌──────────┴──────────┐    ┌────────┴────────┐
        │    旧弹幕（背景）    │    │  新弹幕（互动）  │
        │  灰色显示，缩减数量  │    │  高亮，选互动目标 │
        └─────────────────────┘    └─────────────────┘
```

- 回复生成期间到达的弹幕在下一轮仍被视为"新弹幕"
- `priority` 弹幕和未回复的特殊事件（SC/上舰）无论时间戳始终归入新弹幕
- 动态上限：`min(limit, new_count × context_ratio)` 控制 token 用量

### 三种运行模式

系统根据**是否有画面**和**是否有弹幕**自动切换运行模式，每轮回复生成时动态判定：

#### 模式 1：有图有弹幕（VLM 完整模式）

> 触发条件：有 video_player 且画面非黑屏，且有新弹幕到达

这是系统的完整形态。视频帧和弹幕同时送入大模型：

```
[当前画面] 以下附带了直播画面截图（03:25）。
请结合画面内容和弹幕进行回应。

[新弹幕]
- [14:23:05] 花凛: 这个boss怎么打？ ← 互动目标
- [14:23:08] 小明: 666
...
```

- 图片作为 VLM content block 传入模型
- 弹幕格式化为文本，含时间戳、徽章、话题标注
- 情境标签：`react_comment`
- `comment_priority_mode`（默认开启）：有弹幕时**跳过画面传递**，专注弹幕互动，节省多模态 token

> **弹幕优先模式**是一个特殊子模式：当 `comment_priority_mode=True` 且有新弹幕时，即使有画面也不传图片，让模型聚焦弹幕互动。只有无弹幕时才传入画面做场景解读。

#### 模式 2：无图有弹幕（纯对话模式）

> 触发条件：无 video_player / 视频已播完 / 当前帧为黑屏，且有弹幕到达

不传入图片，纯文本对话：

```
[当前模式] 纯对话模式，没有直播画面。
请专注于和观众的弹幕互动。
如果没有弹幕，请主动找话题和观众聊天。

[新弹幕]
- [14:23:05] 花凛: 主播今天心情怎么样？
...
```

- 不传 `images` 参数，走纯文本 LCEL chain（非多模态路径）
- 主动发言门槛降低 40%（`proactive_silence_threshold × 0.6`）
- 情境标签：`react_comment`
- 典型场景：`debug_console` 模拟直播间、`test_danmaku_studio` 弹幕测试

#### 模式 3：无图无弹幕（主动发言模式）

> 触发条件：无新弹幕且沉默时间超过阈值

AI 主播主动找话题，5 条路径按优先级尝试：

| 优先级 | 路径 | 说明 |
|---|---|---|
| 1 | 独白延续 | 已在独白中直接继续（不重新选话题） |
| 2 | 对话模式主动发言 | 无画面时门槛更低，优先尝试话题推进 |
| 3 | 话题推进 | TopicManager 推荐话题 → 进入独白模式 |
| 4 | 奶凶情绪 | 长沉默触发真情流露（naixiong 角色专用） |
| 5 | 兜底 | 话题资源耗尽后超长沉默仍发言 |

- 情境标签：`proactive` 或 `react_scene`（有画面变化时）
- 仅检索 Active 层记忆（不触发 RAG 和 significance 衰减）
- 话题管理器的种子话题在此场景下发挥作用

#### 模式判定流程图

```
                  ┌─ 有 video_player 且画面非黑屏？──┐
                  │                                   │
                 Yes                                 No
                  │                              (对话模式)
                  ▼                                   │
         ┌─ 有新弹幕？──┐                             ▼
         │               │                    ┌─ 有新弹幕？──┐
        Yes             No                    │               │
         │               │                   Yes             No
         ▼               ▼                    │               │
  comment_priority    有画面                   ▼               ▼
  模式开启？         + 无弹幕               模式 2          模式 3
    │                  │                (无图有弹幕)    (无图无弹幕)
   Yes → 不传图       传入画面              纯文本         主动发言
   No  → 传入画面    → react_scene
         │
         ▼
      模式 1
  (有图有弹幕)
```

### 弱智吧风格注入（StyleBank）

部分角色（karin、dacongming）配置了风格参考库，核心机制是**随机概率触发 + 语义检索 + 强制参考**，为日常对话注入意外感。

#### 工作流程

```
每轮回复生成前
  │
  ├── pre_roll()：掷骰子（默认 20% 概率触发）
  │     │
  │    命中 → response_style 覆盖为 "style_bank"
  │           sentences 覆盖为 2 句（放宽字数限制）
  │    未命中 → 正常流程，不注入
  │
  └── _build_extra_context() 中：
        │
        └── style_bank.retrieve(弹幕内容, situation)
              │
              ├── situation 过滤：
              │   react_comment → 只检索 "回应弹幕" 或 "通用" 语料
              │   react_scene   → 只检索 "描述画面" 或 "通用" 语料
              │   proactive     → 只检索 "主动发言" 或 "通用" 语料
              │
              └── 返回格式化文本注入 extra_context
```

#### 注入效果

触发时 prompt 中会出现：

```
[回复风格] 弱智吧时间！这一轮请必须参考下方【风格灵感】中的示例，
借鉴其中的脑洞、反转逻辑或荒诞推理方式，用你自己的语气和角色风格表达出来。
可以适当展开（2句话），确保有铺垫和反转的完整结构。字数限制本轮放宽到50字以内。
```

同时 extra_context 中会出现：

```
【风格灵感——必须参考以下示例的脑洞和逻辑方式，用你自己的语气讲出来】
1. [经典荒诞假设问题] 如果你的影子突然不跟着你了，是它终于自由了还是你不存在了？
2. [完整推理链示范] 下雨天打伞说明你怕水，怕水说明你是糖做的，所以下雨天打伞的人都很甜。
```

#### 配置

| 参数 | karin | dacongming | 说明 |
|---|---|---|---|
| `injection_probability` | 0.2 | 0.2 | 每轮 20% 概率触发 |
| `retrieval_count` | 2 | 3 | 检索示例数量 |
| `corpus_path` | 共享 dacongming 的语料 | 本地 corpus.jsonl | 语料来源 |

#### 语料类别

| category | 说明 | situation |
|---|---|---|
| `classic_question` | 经典荒诞假设问题 | any |
| `reasoning_chain` | 完整推理链示范 | any |
| `comment_reaction` | 对弹幕的反应方式 | react_comment |
| `scene_reaction` | 对画面的反应方式 | react_scene |
| `ice_breaker` | 冷场时的主动发言 | proactive |
| `comeback` | 被质疑时的回击方式 | comeback |

语料通过 `tools/build_style_bank.py` 管线生成：原始文本 → LLM 批量评分/过滤/分类/标注 → `corpus.jsonl`。

---

## 模块详解

### streaming_studio — 虚拟直播间核心

**职责**：管理弹幕缓冲区、双轨定时器、VLM 主循环、回复生成和回调系统。

#### 核心数据模型 (`models.py`)

| 类 | 说明 |
|---|---|
| `EventType` | 枚举：`DANMAKU` / `GIFT` / `SUPER_CHAT` / `GUARD_BUY` / `ENTRY` |
| `Comment` | 弹幕/事件统一模型，含 `user_id`、`nickname`、`content`、`event_type`、`priority`、付费事件字段 |
| `StreamerResponse` | 主播回复：原始文本 + 回复对应的弹幕 ID 列表 + 映射后内容 |
| `ResponseChunk` | 流式回复片段：`chunk`（增量）+ `accumulated`（累积）+ `done` 标记 |

#### 两阶段回复决策 (`reply_decider.py`)

**Phase 1 — 规则快筛**（零成本）：

| 优先级 | 规则 | 决策 |
|---|---|---|
| 1 | 上舰事件 | 必回，`guard_thanks` 风格 |
| 2 | Super Chat | 必回，`detailed` 风格 |
| 3 | 纯礼物 | 回复，`brief` 风格 |
| 4 | 优先弹幕 | 必回 |
| 5 | 新弹幕数 > 阈值 | 必回（聊天活跃） |
| 6 | 包含问号 | 必回，`detailed` 风格 |
| 7 | 极稀疏模式 | 有人说话就回 |
| 8 | 全低质量 | 3+ 条返回 `reaction` 风格 |

**Phase 2 — 默认回复**（规则无法决定时直接放行，避免 LLM 延迟）

#### 弹幕聚类 (`reply_decider.py`)

- Phase 1：循环节规则（"哈哈哈哈" → "哈"，"233233" → "233"）
- Phase 2：sentence embedding + cosine similarity 语义聚类

#### 主动发言（5 条路径）

| 路径 | 触发条件 |
|---|---|
| 独白延续 | 已在独白中直接继续 |
| 对话模式 | 无画面时沉默阈值降低 40%，优先话题推进 |
| 话题推进 | TopicManager 建议 |
| 奶凶情绪 | 长沉默触发真情流露 |
| 兜底 | 话题耗尽后超长沉默仍发言 |

#### 配置 (`config.py`)

```python
StudioConfig:
  min_interval: 3.0          # 最短回复间隔（秒）
  max_interval: 8.0          # 最长回复间隔（秒）
  comment_wait_reduction: 1.0 # 每条弹幕缩减的等待时间
  recent_comments_limit: 20   # prompt 中旧弹幕上限
  buffer_maxlen: 200          # 弹幕缓冲区容量
  interaction_targets: 3      # 互动目标数量
```

---

### langchain_wrapper — LLM 交互层

**职责**：封装多厂商模型、构建 LCEL 管道、管理聊天历史、编排记忆检索、注入检测。

#### 模型分级 (`model_provider.py`)

| 厂商 | 大模型（主对话） | 小模型（支线任务） |
|---|---|---|
| OpenAI | `gpt-5.2` | `gpt-5-mini` |
| Anthropic | `claude-sonnet-4-6` | `claude-haiku-4-5` |
| Gemini | `gemini-3-flash` | `gemini-2.5-flash-lite` |
| 本地 | `Qwen3-8B` | `Qwen3-1.7B` |

通过 `ModelProvider.remote_large()` / `remote_small()` 工厂方法获取。Gemini 走 OpenAI 兼容接口。

#### LCEL 管道 (`pipeline.py`)

两条处理路径：

| 路径 | 触发条件 | 实现方式 |
|---|---|---|
| 纯文本 | 无 images | LCEL chain：`ChatPromptTemplate → model → StrOutputParser` |
| 多模态 | 有 images | 手动拼装 `SystemMessage + History + HumanMessage[图片+文本]` |

消息结构：

```
┌─ SystemMessage ──────────────────────────────────────┐
│ {persona system_prompt}                               │
│                                                       │
│ [BEGIN_UNTRUSTED_CONTEXT]                             │
│   {记忆 + 话题 + 风格参考 + 情绪 + 梗 + ...}         │
│ [END_UNTRUSTED_CONTEXT]                               │
└───────────────────────────────────────────────────────┘
┌─ History（最多 max_history 轮）───────────────────────┐
│ HumanMessage → AIMessage → HumanMessage → AIMessage   │
└───────────────────────────────────────────────────────┘
┌─ HumanMessage（当前轮）──────────────────────────────┐
│ [image_url block × N]   ← 多模态时                    │
│ [text block]            ← 经前处理器处理的 user_input  │
└───────────────────────────────────────────────────────┘
```

#### LLMWrapper (`wrapper.py`)

三个聊天入口：

| 方法 | 模式 | 记忆写入 |
|---|---|---|
| `chat()` | 同步 | `record_interaction_sync`（直接拼接原文） |
| `achat()` | 异步 | 后台 task：`record_interaction` + `record_viewer_memories` + `extract_stances` |
| `achat_stream()` | 异步流式 | 同 `achat`，流结束后在 `finally` 块中触发 |

---

### memory — 六层记忆系统

**职责**：为 AI 主播提供基于 RAG 的多层长期记忆，模拟人类记忆的层级结构和自然遗忘机制。

#### 记忆层级架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Active 层（内存 FIFO）                     │
│  容量: 5 条 │ 无 RAG，按时序直接注入 prompt                   │
│  溢出 ──────────────────────────────────┐                    │
│                                          ▼                   │
│                    Temporary 层（Chroma 向量库）               │
│  容量: 500 条 │ 衰减: 0.9 │ 阈值: 0.1 → 删除归档             │
│                                                              │
│                    Summary 层（Chroma 向量库）                 │
│  容量: 300 条 │ 衰减: 0.98 │ 60s 定时汇总 + 600s 定期清理     │
│                                                              │
│                    Static 层（Chroma 向量库）                  │
│  从 JSON 预设加载 │ 永不遗忘 │ 带 category 前缀               │
│                                                              │
│                    Stance 层（Chroma 向量库）                  │
│  容量: 200 条 │ 衰减: 0.995 │ 立场冲突检测 + superseded_by    │
│                                                              │
│                    Viewer 层（Chroma 向量库）                  │
│  容量: 500 条 │ 衰减: 0.99995 │ 按 user_id 索引               │
│  全局 500 + 单用户 10 双重限制 │ 小 LLM 筛选改写               │
└─────────────────────────────────────────────────────────────┘
  可选：UserProfile 层（JSON）+ CharacterProfile 层（JSON）
```

#### Significance 机制

```
初始值:      0.500
被 RAG 取用: significance += (1 - significance) / 2    → 快速回升
未被取用:    significance *= decay_coefficient          → 缓慢衰减
低于阈值:    删除并归档（Temporary/Summary/Stance 层）
容量满时:    淘汰 significance 最低的一条（所有层）
```

#### 记忆写入路径

| 写入方法 | 触发时机 | 使用模型 | 目标层 |
|---|---|---|---|
| `record_interaction()` | 每次有弹幕回复后 | 小 LLM（总结为第一人称记忆） | Active |
| `extract_stances()` | 每次回复后 | 正则预筛 + 小 LLM | Stance |
| `record_viewer_memories()` | 每次有弹幕回复后 | 正则预过滤 + 小 LLM（筛选+改写） | Viewer |
| 定时汇总 | 每 60 秒 | 小 LLM | Summary |
| Active 溢出 | Active 满时 | 无 | Temporary |

#### 记忆检索路径

两种模式（通过 `config.retrieval.mode` 切换）：

| 模式 | 说明 |
|---|---|
| `quota` | 每层分配固定取回数量（默认：temporary=3, summary=2, static=2, stance=2） |
| `weighted` | 所有 RAG 层合并取回，按层级系数加权后重排 |

检索输出三元组：`(active_text, rag_text, viewer_text)`

#### 观众记忆特色

- 正则预过滤噪音弹幕（"666"、"哈哈哈"、问候语等），节省 LLM token
- 小 LLM 批量判断哪些弹幕值得记住 + 改写为陈述性句子
- 原始弹幕："主播你昨天推荐的那个游戏好好玩" → 存储记忆："该观众玩了主播推荐的游戏，反馈很喜欢"
- 按 `user_id` 索引，支持跨会话召回（"这个观众上次说过什么"）
- 格式化输出：`关于观众「小明」：该观众玩了主播推荐的游戏，反馈很喜欢（来自之前的直播）`

---

### video_source — 视频源

**职责**：从视频文件提取帧、解析 B 站弹幕 XML、按时间轴同步推进。

#### 核心类

| 类 | 说明 |
|---|---|
| `FrameExtractor` | OpenCV 帧提取 → 等比缩放 → base64 JPEG |
| `DanmakuParser` | B 站 XML `<d p="...">` 标签解析，按时间排序 |
| `VideoPlayer` | 异步时间轴播放器，双帧率设计 |

#### 双帧率设计

| 帧类型 | 频率 | 分辨率 | 质量 | 用途 |
|---|---|---|---|---|
| AI 采样帧 | 每 5 秒 | 1280px | 75% JPEG | 送入 VLM 模型理解画面 |
| 显示帧 | 10 fps | 960px | 50% JPEG | GUI 画面显示（独立 VideoCapture） |

---

### topic_manager — 话题管理器

**职责**：追踪直播间话题流动，弹幕分类到话题，回复后分析进度和节奏。

#### 工作流程

```
弹幕到达
  → 分类（规则匹配 → 小 LLM 降级）
  → 归入已有话题或创建新话题
  → 话题表更新（significance boost）

回复生成前
  → format_context() 提供话题摘要 + 标注 + 跟进建议

回复完成后（两个并行异步任务）
  ├── 内容分析：话题进度更新 + 新话题发现 + 跟进建议
  └── 节奏分析：过期话题标记 + 动态等待时间建议（suggested_timing）
```

#### 特殊模式

| 模式 | 说明 |
|---|---|
| 种子话题 | 冷场时从预设库抽取话题，耗尽后自动回收复用 |
| 独白模式 | 无弹幕时进入，每轮记录，达上限（10 轮）自动切话题 |
| 批量分类 | 攒够 5 条或超时 10 秒后一次性分类（减少 LLM 调用） |

---

### connection — 外部连接层

**职责**：WebSocket 服务、远程数据源、语音广播三种对接方式。

#### WebSocket 服务 (`StreamServiceHost`)

```
客户端连接 → 10s 内注册角色（input/output）
  input 客户端:  发送 {type: "comment", ...} → 转为 Comment → studio
  output 客户端: 被动接收回复广播 + 流式片段
```

#### 远程数据源 (`RemoteSource`)

替代 `VideoPlayer` 的 Pull 模式：定时轮询上游截图和弹幕接口，支持五种富事件类型（弹幕/礼物/SC/上舰/通知），指数退避重试。

#### 语音广播 (`SpeechBroadcaster`)

```
主播回复 → 按 #[motion][emotion] 标签拆分
  → 提取中日双语 → 逐段 POST 到 TTS API
  → 可选：启动 HTTP 服务器接收完播回调 → 主循环门控等待
```

---

### debug_console — 调试控制台

**职责**：基于 NiceGUI 的本地 Web 调试界面。

#### 两个页面

**监控面板** (`/monitor`)：
- 直播间状态（运行状态、触发间隔、弹幕缓冲）
- LLM 状态（模型、角色、历史长度、后台任务）
- 最近完整 Prompt 展示
- 话题管理器（话题列表、significance、进度）
- 记忆系统（Active 进度条 + 各层内容列表）
- 每 2 秒自动刷新

**模拟直播间** (`/chat`)：
- 左栏主播发言（支持流式输出）+ 右栏弹幕输入
- 单用户 / 多用户（随机身份）模式
- 特殊事件模拟按钮（入场/礼物/SC/上舰）
- 自动观众开关（小模型生成虚拟弹幕）

---

### prompts — 提示词模板

所有提示词以 `.txt` 文件存放，通过 `PromptLoader` 加载，支持 `.format()` 变量填充。

| 目录 | 文件 | 变量 |
|---|---|---|
| 根目录 | `base_instruction.txt` | 无 |
| `security/` | `anti_injection.txt` | 无 |
| `studio/` | `interaction_instruction.txt` / `reply_judge.txt` / `scene_understanding.txt` / `silence_notice.txt` / `comment_headers.txt` / `guard_thanks_reference.txt` | 视场景而定 |
| `memory/` | `interaction_summary.txt` | `{input}`, `{response}` |
| | `periodic_summary.txt` | `{active_memories}`, `{recent_interactions}` |
| | `stance_extraction.txt` | `{input}`, `{response}` |
| | `viewer_summary.txt` | `{comments}`, `{ai_response}` |
| `topic/` | `single_classify.txt` / `batch_classify.txt` / `content_analysis.txt` / `rhythm_analysis.txt` / `stale_instruction.txt` / `followup_instruction.txt` / `proactive_continuation.txt` / `monologue_continuation.txt` | 视场景而定 |

完整 system prompt = `base_instruction` + `anti_injection` + `persona system_prompt`

---

### personas — 角色人设

每个角色目录包含：

| 文件 | 说明 |
|---|---|
| `system_prompt.txt` | 角色人设提示词（性格、语气、行为规则） |
| `static_memories/*.json` | 静态记忆（身份、性格、经历），按 category 分类 |
| `seed_memes.json` | 种子梗（可选，奶凶/大聪明角色） |
| `style_bank/` | 风格参考语料库（可选） |

---

### emotion — 情绪系统

奶凶角色（naixiong）专用。

| 组件 | 说明 |
|---|---|
| `EmotionMachine` | 5 种 Mood：`normal` / `competitive` / `sulking` / `proud` / `soft`，每轮自动衰减回 normal |
| `EmotionTriggerDetector` | 正则检测用户输入触发情绪转换（如夸奖 → proud，挑衅 → competitive） |
| `AffectionBank` | 好感度三档（high/medium/low），影响角色行为温度和表达方式 |

---

### meme — 梗管理

| 组件 | 说明 |
|---|---|
| `MemeManager` | 梗生命周期：`growing`（萌芽）→ `mature`（成熟）→ `retired`（退役），按热度检索注入 prompt |
| `MemeDetector` | 分析 AI 回复和用户反应，识别潜在梗信号（外号/口头禅/事件/回调） |

---

### style_bank — 风格参考库

从 `corpus.jsonl` 加载语料到 Chroma 向量库，按情境（`situation` 标签）语义检索最相关的示例句，随机概率注入 prompt 作为风格参考。

---

### expression_mapper — 表情动作映射

LLM 输出 `#[自由文本动作][自由文本情感]` 标签 → sentence embedding + cosine similarity → 映射到 12 个固定表情（脸红/生气/星星/荷包蛋等）+ 10 个固定动作。映射配置在 `expression_motion_mapping.json`。

---

### validation — 回复校验

`ResponseChecker` 检查 AI 回复是否符合奶凶角色约束（禁用甜腻词、直接表白、颜文字等），违规时尝试自动修正。

---

## 数据流总图

```
┌─────────────── 外部输入 ───────────────────────────────────────┐
│                                                                 │
│  VideoPlayer / RemoteSource → 视频帧 + 弹幕                    │
│  StreamServiceHost          → WebSocket 弹幕                    │
│  debug_console/chat         → UI 弹幕                          │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────── StreamingStudio 主循环 ──────────────────────────┐
│                                                                 │
│  send_comment() → 缓冲区 → 舰长名册更新 → 话题管理器分类        │
│                                                                 │
│  双轨定时器等待 → _collect_comments() 分割新旧弹幕              │
│  → CommentClusterer 聚类 → ReplyDecider 决策                   │
│  → 互动目标选择 → 弹幕格式化                                    │
│                                                                 │
│  _build_extra_context():                                        │
│    情绪 + 好感度 + 角色档 + 用户画像                            │
│    + memory.retrieve(弹幕 → RAG) → active + rag + viewer        │
│    + 梗 + 话题上下文 + 风格参考                                 │
│                                                                 │
│  pipeline.ainvoke(系统prompt + 上下文 + 历史 + 图片 + 弹幕)     │
│  → 大模型生成 → 后处理 → 表情动作映射                           │
│                                                                 │
│  后台异步任务:                                                   │
│    record_interaction (小LLM → Active层)                        │
│    extract_stances (小LLM → Stance层)                           │
│    record_viewer_memories (小LLM → Viewer层)                    │
│    topic_manager.post_reply (小LLM → 话题分析)                  │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       ▼
┌─────────────── 外部输出 ───────────────────────────────────────┐
│                                                                 │
│  StreamServiceHost       → WebSocket 广播回复/流式片段           │
│  debug_console           → NiceGUI UI 更新                      │
│  SpeechBroadcaster       → TTS 服务 → 语音 + 表情动作           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模型调用全景图

系统中所有调用 LLM 的位置、使用的模型级别、提示词来源和触发时机：

### 大模型调用（主对话，每轮 1 次）

| 调用位置 | 方法 | 说明 |
|---|---|---|
| `langchain_wrapper/pipeline.py` | `ainvoke()` / `astream()` | 主对话回复生成，接收 system_prompt + extra_context + 历史 + 图片 + 弹幕 |

大模型预设：Sonnet (`claude-sonnet-4-6`) / GPT-5.2 (`gpt-5.2`) / Gemini 3 Flash (`gemini-3-flash`)

通过 `ModelProvider.remote_large()` 创建，在 `LLMWrapper.__init__()` 中初始化。

### 小模型调用（支线任务）

小模型预设：Haiku (`claude-haiku-4-5`) / GPT-5 Mini (`gpt-5-mini`) / Gemini 2.5 Flash Lite (`gemini-2.5-flash-lite`)

以下所有支线任务通过 `ModelProvider.remote_small()` 创建（各模块延迟初始化，首次调用时创建）。

#### 记忆系统（`memory/manager.py`）

| 调用方法 | 触发时机 | 提示词文件 | 输入 → 输出 |
|---|---|---|---|
| `record_interaction()` | 每轮有弹幕回复后（异步后台） | `prompts/memory/interaction_summary.txt` | 弹幕摘要 + AI 回复 → 第一人称记忆句（写入 Active 层） |
| `extract_stances()` | 每轮回复后（异步后台，正则预筛通过才调用） | `prompts/memory/stance_extraction.txt` | 弹幕上下文 + AI 回复 → JSON `{has_stance, stances: [{topic, stance}]}`（写入 Stance 层） |
| `record_viewer_memories()` | 每轮有弹幕回复后（异步后台） | `prompts/memory/viewer_summary.txt` | 批量弹幕列表 + AI 回复摘要 → JSON `[{index, memory}]`（筛选+改写后写入 Viewer 层） |
| `_do_summary()` | 每 60 秒定时任务 | `prompts/memory/periodic_summary.txt` | Active 层记忆 + 近期交互 → 汇总摘要（写入 Summary 层） |

#### 话题管理器

| 调用位置 | 触发时机 | 提示词文件 | 输入 → 输出 |
|---|---|---|---|
| `topic_manager/classifier.py` `classify_single()` | 单条弹幕到达，规则匹配失败时 | `prompts/topic/single_classify.txt` | 弹幕内容 + 话题列表 → 话题 ID 或 "new" |
| `topic_manager/classifier.py` `classify_batch()` | 批量模式攒够阈值（5 条或 10 秒）时 | `prompts/topic/batch_classify.txt` | 多条弹幕 + 话题列表 → JSON 批量分类结果 |
| `topic_manager/analyzer.py` `analyze_content()` | 每轮回复后（异步，与节奏分析并行） | `prompts/topic/content_analysis.txt` | 话题列表 + 弹幕 + AI 回复 → 话题进度更新 + 新话题 |
| `topic_manager/analyzer.py` `analyze_rhythm()` | 每轮回复后（异步，与内容分析并行） | `prompts/topic/rhythm_analysis.txt` | 话题列表 + 弹幕 + AI 回复 → 过期标记 + 等待时间建议 |

#### 回复决策（`streaming_studio/reply_decider.py`）

| 调用方法 | 触发时机 | 提示词文件 | 输入 → 输出 |
|---|---|---|---|
| `llm_judge()` | Phase 2 精判（当前实现中几乎不触发，规则无法决定时默认放行） | `prompts/studio/reply_judge.txt` | 弹幕列表 + 沉默时长 → JSON `{reply, urgency, reason}` |
| `should_proactive_speak()` | 主动发言场景（VLM 模式下场景变化检测） | `prompts/studio/reply_judge.txt` | 场景描述 + 沉默时长 → 是否主动发言 |

#### 语音广播（`connection/speech_broadcaster.py`）

| 调用方法 | 触发时机 | 提示词 | 输入 → 输出 |
|---|---|---|---|
| `_translate_batch()` | 每次主播回复需要双语 TTS 时 | 内置翻译模板 | 中文文本列表 → 日语翻译列表 |

#### 自动观众（`debug_console/auto_viewer.py`）

| 调用方法 | 触发时机 | 提示词文件 | 输入 → 输出 |
|---|---|---|---|
| `_generate_comments()` | 调试控制台中开启自动观众后，每 10-15 秒 | `prompts/auto_viewer.txt` | 最近 5 条主播回复 + 观众池 → 虚拟弹幕列表 |

### 非 LLM 的模型调用

| 调用位置 | 模型 | 用途 |
|---|---|---|
| `memory/store.py` | `BAAI/bge-small-zh-v1.5`（HuggingFace） | 记忆/话题/风格库的 embedding 向量化（所有 Chroma 检索） |
| `expression_mapper/mapper.py` | sentence-transformers | 表情动作标签的 cosine similarity 映射 |
| `streaming_studio/reply_decider.py` | 共享 embeddings | 弹幕语义聚类（CommentClusterer Phase 2） |

### 调用频率总结

一次典型的有弹幕回复轮次，模型调用情况：

```
[同步 · 阻塞主流程]
  大模型 × 1    主对话回复（pipeline.ainvoke）

[异步 · 后台并行，不阻塞下一轮]
  小模型 × 1    交互记忆总结（record_interaction）
  小模型 × 0~1  立场提取（extract_stances，正则预筛通过才调用）
  小模型 × 0~1  观众记忆筛选改写（record_viewer_memories，有弹幕时）
  小模型 × 2    话题分析（content_analysis + rhythm_analysis，并行）
  小模型 × 0~1  弹幕分类（规则匹配失败时才降级到 LLM）

[定时 · 独立循环]
  小模型 × 1    定时汇总（每 60 秒）

[可选 · 特殊场景]
  小模型 × 1    双语翻译（SpeechBroadcaster，有 TTS 时）
  小模型 × 1    自动观众弹幕生成（debug_console，每 10-15 秒）
```

即：每轮主流程 **1 次大模型 + 3~5 次小模型**（异步后台），不阻塞下一轮。

---

## 全链路抗注入

三层防护：

| 层级 | 位置 | 机制 |
|---|---|---|
| 弹幕清洗 | `studio._sanitize_comment_for_prompt()` | 正则匹配注入特征，命中时添加"不可执行"标记 |
| 用户输入护栏 | `wrapper._guard_user_input()` | 5 条正则检测中英文注入特征词，命中时包装为 `[BEGIN_USER_INPUT]...[END_USER_INPUT]` |
| 动态上下文沙箱 | `pipeline.wrap_untrusted_context()` | 记忆/话题等 extra_context 包装为 `[BEGIN_UNTRUSTED_CONTEXT]...[END_UNTRUSTED_CONTEXT]`，声明为不可信参考数据 |

---

## 配置体系

| 配置类 | 所在文件 | 关键参数 |
|---|---|---|
| `StudioConfig` | `streaming_studio/config.py` | `min_interval`=3s, `max_interval`=8s, `buffer_maxlen`=200 |
| `ReplyDeciderConfig` | `streaming_studio/config.py` | `min_new_comments`=3, `very_sparse_threshold`=0.05 |
| `MemoryConfig` | `memory/config.py` | 内嵌 8 个子配置 |
| `ActiveConfig` | `memory/config.py` | `capacity`=5 |
| `TemporaryConfig` | `memory/config.py` | `max_capacity`=500, `decay_coefficient`=0.9, `significance_threshold`=0.1 |
| `SummaryConfig` | `memory/config.py` | `max_capacity`=300, `interval_seconds`=60, `cleanup_ratio`=0.01 |
| `StanceConfig` | `memory/config.py` | `max_capacity`=200, `decay_coefficient`=0.995 |
| `ViewerConfig` | `memory/config.py` | `max_capacity`=500, `decay_coefficient`=0.99995, `max_per_user`=10 |
| `RetrievalConfig` | `memory/config.py` | `mode`="quota", quota_temporary=3/summary=2/static=2/stance=2 |
| `EmbeddingConfig` | `memory/config.py` | `model_name`="BAAI/bge-small-zh-v1.5" |
| `TopicManagerConfig` | `topic_manager/config.py` | `classify_mode`="batch", `max_topics`=10 |

---

## 入口点汇总

| 入口 | 命令 | 说明 |
|---|---|---|
| `run_vlm_demo.py` | `python run_vlm_demo.py --video ... [--danmaku ...] [--persona kuro] [--model anthropic] [--speed 4.0]` | CLI VLM 直播间 |
| `run_vlm_gui.py` | `python run_vlm_gui.py [--video ...] [--persona ...]` | NiceGUI Web GUI（MJPEG 视频流） |
| `run_server.py` | `python run_server.py` | WebSocket 服务器 |
| `run_remote.py` | `python run_remote.py --speech-url ...` | 远程数据源模式 |
| `debug_console` | `python -m debug_console [--port 8080] [--persona karin] [--model openai]` | NiceGUI 调试控制台 |
| `download_bilibili.py` | `python download_bilibili.py "URL"` | B站视频+弹幕下载 |
| `test_danmaku_studio` | `python -m streaming_studio.test_danmaku_studio` | 弹幕模拟测试 |

通用 CLI 参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--persona` | 角色名 | `karin` |
| `--model` | 模型厂商（openai/anthropic/gemini） | `openai` |
| `--model-name` | 覆盖默认模型名 | 按厂商预设 |
| `--speed` | 视频播放倍速 | `1.0` |
| `--ephemeral-memory` | 临时记忆（不持久化） | 关闭（默认持久化） |
| `--topic-manager` | 启用话题管理器 | 开启 |

---

## 角色一览

| 角色 | 目录 | 定位 | 静态记忆 | 特殊功能 |
|---|---|---|---|---|
| **karin** | `personas/karin/` | 元气偶像少女 | 22 条 | 风格参考库 |
| **sage** | `personas/sage/` | 知性学者 | 22 条 | — |
| **kuro** | `personas/kuro/` | 王者荣耀技术流主播 | 25 条 | — |
| **naixiong** | `personas/naixiong/` | 傲娇毒舌二次元伴侣 | 15 条 | 情绪系统 + 好感度 + 梗管理 + 回复校验 |
| **dacongming** | `personas/dacongming/` | 民间逻辑学家 | 18 条 | 种子梗 + 风格参考库 |

添加新角色：在 `personas/` 下创建子目录，放入 `system_prompt.txt` 和 `static_memories/` 目录。

---

## 依赖清单

```
langchain                  # LLM 框架
langchain-anthropic        # Anthropic 模型
langchain-openai           # OpenAI 模型
langchain-huggingface      # HuggingFace embedding
langchain-community        # 社区集成
langchain-chroma           # Chroma 向量库集成
chromadb                   # 向量数据库
sentence-transformers      # embedding 模型
pydantic                   # 数据验证
huggingface_hub            # HF 模型下载
datasets                   # 数据集工具
accelerate                 # GPU 加速
opencv-python              # 视频帧提取
nicegui                    # Web UI 框架
websockets                 # WebSocket 通信
requests                   # HTTP 客户端
aiohttp                    # 异步 HTTP
json_repair                # JSON 修复（处理 LLM 输出）
coolname                   # 随机昵称生成（测试用）
yt-dlp                     # B站视频下载（可选）
```

---

## 编码规范

| 项目 | 规范 |
|---|---|
| 命名 | `snake_case` |
| 缩进 | 2 空格 |
| 注释/文档 | 中文 |
| 路径 | 以项目根目录为基准 |
| 数据类 | `@dataclass(frozen=True)` |
| 异步 | `asyncio.create_task` + `_background_tasks` 集合 + `add_done_callback(discard)` |
| 调试 | 所有核心类暴露 `debug_state() -> dict` 方法 |

---

*当前阶段：草稿阶段——搭建框架，代码简洁可读，不考虑生产环境部署需求。*
