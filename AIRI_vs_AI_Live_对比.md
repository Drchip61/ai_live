# AIRI vs AI Live 对比分析

## 1. 项目概览

| 维度 | AIRI | AI Live |
|------|------|---------|
| **定位** | 自托管 LLM 虚拟伴侣平台，目标是开源复刻 Neuro-sama | VLM 虚拟直播间系统，AI 主播同时理解画面和弹幕 |
| **口号** | "让你拥有自己的数字生命" | — |
| **核心场景** | VTuber / 数字伴侣 / 游戏 Agent | 视频直播间（画面理解 + 弹幕互动） |
| **开源协议** | MIT | — |
| **社区规模** | 28.3k Stars, 2.7k Forks, 23+ 贡献者 | 个人项目 |
| **开发阶段** | v0.8.4+，活跃迭代中 | 草稿阶段，专注框架搭建 |
| **主要语言** | 中文 + 英文 + 日文 | 中文 |

---

## 2. 技术栈对比

| 维度 | AIRI | AI Live |
|------|------|---------|
| **主语言** | TypeScript + Rust (Tauri 插件) | Python |
| **框架** | Vue 3 + Vite, Hono/h3 (后端), Electron, Capacitor | LangChain, NiceGUI (调试面板) |
| **包管理** | pnpm Workspaces + Turbo (monorepo) | pip + requirements.txt |
| **LLM 抽象** | xsAI (自研，20+ Provider 统一接口) | LangChain (Anthropic/OpenAI/Gemini) |
| **向量数据库** | PostgreSQL + pgvector / DuckDB WASM | Chroma (chromadb) |
| **Embedding** | BAAI/bge-small-zh-v1.5 | BAAI/bge-small-zh-v1.5 (相同) |
| **视频处理** | Three.js / Live2D SDK (渲染虚拟形象) | OpenCV (真实视频帧提取) |
| **音频** | ElevenLabs / 阿里云 / 火山引擎 / Whisper | SpeechBroadcaster (外部 TTS 服务对接) |
| **数据库** | PostgreSQL, DuckDB | SQLite (弹幕存储), Chroma (记忆) |
| **运行时** | Node.js 24.9+ / Electron / 浏览器 | Python asyncio |

---

## 3. 架构对比

### AIRI — 大型 Monorepo

```
airi/
├── apps/              # 3 个可部署应用
│   ├── web/           # 浏览器 PWA (Cloudflare Workers)
│   ├── tamagotchi/    # 桌面应用 (Electron)
│   └── pocket/        # 移动端 (Capacitor)
├── packages/          # 40+ 共享库
│   ├── stage-ui/      # UI 组件 + 状态管理
│   ├── core-character/# 角色灵魂容器
│   ├── audio/         # 语音管线 (VAD/STT/TTS)
│   └── ...
├── services/          # 独立 Bot 服务
│   ├── discord-service/
│   ├── telegram-service/
│   ├── minecraft-agent/
│   └── factorio-service/
└── crates/            # Rust Tauri 插件
```

- **架构风格**：微服务 + 共享库 Monorepo
- **通信方式**：WebSocket (全双工)
- **构建编排**：Turbo 依赖图构建

### AI Live — Python 单体

```
ai_live/
├── streaming_studio/  # 直播间核心 (主循环、回复决策)
├── langchain_wrapper/ # LLM 层 (模型切换、LCEL 管线)
├── memory/            # 多层记忆系统
├── video_source/      # 视频帧提取 + 弹幕解析
├── topic_manager/     # 话题分类与分析
├── emotion/           # 情绪状态机 + 好感度
├── meme/              # 梗生命周期管理
├── validation/        # 回复一致性检查
├── personas/          # 角色定义
├── prompts/           # 提示词模板
├── debug_console/     # NiceGUI 监控面板
└── connection/        # WebSocket + 语音广播
```

- **架构风格**：全异步单体，模块间直接调用
- **数据流**：弹幕 + 视频帧 → 记忆检索 → VLM 调用 → 记忆写入
- **后台任务**：`asyncio.Task` 集合 + 自动清理回调

### 对比小结

| 维度 | AIRI | AI Live |
|------|------|---------|
| **代码组织** | 40+ 包的 Monorepo | ~15 个 Python 模块的单体 |
| **部署目标** | 浏览器 + 桌面 + 移动端 + 服务器 | 本地 Python 进程 |
| **复杂度** | 高（跨平台、多运行时） | 中等（单一运行时、聚焦场景） |
| **上手难度** | 较高（Node.js + pnpm + Turbo + 多项目） | 较低（pip install + python run） |

---

## 4. 核心功能对比

### 4.1 记忆系统

| 维度 | AIRI | AI Live |
|------|------|---------|
| **层数** | 4 层 (Active / Temporary / Summary / Static) | 6 层 (Active / Temporary / Summary / Static / **Stance** / **Viewer**) |
| **独有层** | — | Stance（对话题的立场）、Viewer（按用户索引的记忆） |
| **存储后端** | PostgreSQL + pgvector / DuckDB WASM | Chroma 向量库 |
| **检索方式** | 向量语义搜索 | Quota（定额）或 Weighted（加权合并）两种模式 |
| **衰减机制** | 有 (significance decay) | 有 (significance decay，低于阈值自动遗忘) |
| **持久化** | PostgreSQL (生产) / Ephemeral (开发) | 全局持久化 (Chroma DB) / 临时模式 |
| **写入方式** | 小模型总结为第一人称记忆 | 小模型总结为第一人称记忆 (相同) |

### 4.2 LLM 集成

| 维度 | AIRI | AI Live |
|------|------|---------|
| **Provider 数量** | 20+ (via xsAI) | 4 (Anthropic / OpenAI / Gemini / Local Qwen) |
| **抽象层** | xsAI (自研统一接口) | LangChain (ModelProvider 工厂) |
| **大小模型分级** | 有 (大模型对话 + 小模型辅助) | 有 (相同策略) |
| **本地推理** | ONNX Runtime / vLLM / SGLang | Local Qwen 支持 |
| **浏览器推理** | xsai-transformers (实验性) | 不支持 |
| **多模态** | 文本 + 语音为主 | **文本 + 图像** (视频帧直传模型) |

### 4.3 视频 / 视觉能力

| 维度 | AIRI | AI Live |
|------|------|---------|
| **视觉定位** | 虚拟形象渲染 (Live2D / VRM 3D) | 真实视频画面理解 (VLM) |
| **画面理解** | 游戏内 CV (YOLO, MediaPipe) | **视频帧 → 多模态 LLM 理解** |
| **虚拟形象** | Live2D + Three.js VRM (口型/表情同步) | 不含 (聚焦视频内容分析) |
| **帧提取** | 不适用 | OpenCV 可配置间隔 + 分辨率 |

### 4.4 音频 / 语音能力

| 维度 | AIRI | AI Live |
|------|------|---------|
| **TTS** | 多引擎 (ElevenLabs / 阿里云 / 火山 / 浏览器原生) | 外部服务对接 (SpeechBroadcaster) |
| **STT** | Whisper (本地) + 浏览器原生 | 不含 |
| **VAD** | 客户端语音活动检测 | 不含 |
| **口型同步** | Live2D / VRM 实时同步 | 动作标签 (#[motion][emotion]) |
| **语音交互** | 完整双向语音对话 | 单向输出 (AI → 语音) |

### 4.5 弹幕 / 聊天集成

| 维度 | AIRI | AI Live |
|------|------|---------|
| **支持平台** | Discord / Telegram / Twitter | **Bilibili** (弹幕 XML) |
| **交互类型** | 文字聊天 + 语音通话 | 弹幕 / 礼物 / SC / 上舰 / 进场 |
| **优先级系统** | 不明确 | **完整优先级**：舰长 > SC > 礼物 > 弹幕 > 进场 |
| **弹幕分割** | 不适用 | 新旧弹幕分界 + priority 标记 |
| **回复决策** | 不明确 | **两阶段决策**：规则快筛 + LLM 精判 |

### 4.6 角色 / 人设系统

| 维度 | AIRI | AI Live |
|------|------|---------|
| **架构理念** | "灵魂容器" (Soul Container) — 人格为一等公民 | Persona 目录 (system_prompt + static_memories) |
| **配置格式** | YAML/JSON Character Cards + Lorebook | 文本 system_prompt + JSON static_memories |
| **编辑工具** | Web UI 角色卡编辑器 | 手动编辑文件 |
| **预设角色** | 可自定义 | 5 个内置 (karin / sage / kuro / naixiong / dacongming) |
| **条件注入** | 时间/条件触发的 Prompt 注入 | 基于话题和情绪状态的动态注入 |

### 4.7 话题管理

| 维度 | AIRI | AI Live |
|------|------|---------|
| **实现方式** | Lorebook + 条件注入 (if/if-else) | **完整 TopicManager**（分类 + 分析 + 节奏控制） |
| **弹幕分类** | 不适用 | 规则匹配优先 → LLM 降级，单条/批量模式 |
| **话题追踪** | 不明确 | 话题表 (发现 → 活跃 → 过期)，动态调整等待时间 |
| **节奏控制** | 不适用 | `suggested_timing` 动态覆盖回复间隔 |

### 4.8 情绪系统

| 维度 | AIRI | AI Live |
|------|------|---------|
| **情绪状态机** | 通过人设定义 + TTS 情感参数 | **EmotionMachine**：5 种状态 (NORMAL / COMPETITIVE / SULKING / PROUD / SOFT) |
| **好感度系统** | 不明确 | **AffectionBank**：0-100 隐藏分数，三档 (HIGH / MEDIUM / LOW) |
| **自动衰减** | 不明确 | 有 (自动回归 NORMAL) |
| **表达映射** | Live2D/VRM 表情动画 | 颜文字密度控制 + 动作标签 |

### 4.9 其他独有功能

| 功能 | AIRI | AI Live |
|------|------|---------|
| **游戏 Agent** | Minecraft + Factorio 自主游玩 | — |
| **梗管理** | — | **MemeManager**：梗生命周期 (GROWING → MATURE → RETIRED) |
| **回复验证** | — | **ResponseChecker**：违禁词 / 人设一致性 / 颜文字密度检查 |
| **风格库** | — | **StyleBank**：上下文相关的风格示例注入 |
| **注入防护** | 有 (Prompt 清洗) | **三层防护**：弹幕清洗 → 输入护栏 → 上下文沙箱 |
| **主动发言** | 不明确 | 沉默超阈值 + 画面变化 → 自动触发 |
| **调试面板** | 内置 Web UI | NiceGUI 面板 (debug_state 聚合) |
| **多端部署** | 浏览器 / 桌面 / 移动 / 服务器 | 本地 Python 进程 |

---

## 5. 各自优势与互补分析

### AIRI 的优势

1. **跨平台覆盖**：单一代码库部署到浏览器、桌面、移动端，用户触达面广
2. **完整语音管线**：VAD → STT → LLM → TTS → 口型同步，双向语音交互
3. **虚拟形象渲染**：Live2D + VRM 3D 模型，视觉表现力强
4. **多平台社交集成**：Discord / Telegram / Twitter 原生支持
5. **游戏自主体**：Minecraft / Factorio Agent，展现泛化能力
6. **LLM Provider 生态**：20+ 厂商统一接入，灵活切换
7. **社区与生态**：28k+ Stars，活跃的贡献者和版本迭代

### AI Live 的优势

1. **真实视频理解**：VLM 多模态能力，AI 真正"看懂"画面内容 — AIRI 无此能力
2. **直播间深度建模**：弹幕优先级、新旧分界、双轨触发、回复决策 — 专为直播场景优化
3. **六层记忆系统**：Stance（立场）和 Viewer（观众）层是独有设计，记忆检索更精细
4. **话题管理系统**：完整的分类 → 追踪 → 过期 → 节奏控制管线
5. **情绪 + 好感度**：EmotionMachine 状态机 + AffectionBank 隐藏好感，角色表现更有层次
6. **梗生命周期**：自动发现、培育、退休梗，避免重复使用过时笑话
7. **回复质量保障**：ResponseChecker 验证人设一致性，StyleBank 提供风格参考
8. **三层注入防护**：弹幕清洗 → 输入护栏 → 上下文沙箱，安全架构更完善
9. **低复杂度启动**：pip install + 一行命令即可运行

### 互补分析

| AI Live 可借鉴 AIRI | AIRI 可借鉴 AI Live |
|---------------------|---------------------|
| 完整 TTS/STT 语音管线集成 | VLM 视频帧理解能力 |
| Live2D / VRM 虚拟形象渲染 | 弹幕优先级与回复决策系统 |
| 多平台 Bot 服务 (Discord/Telegram) | 话题管理器（分类 + 节奏控制） |
| 跨平台部署 (Web/Desktop/Mobile) | 情绪状态机 + 好感度银行 |
| 更多 LLM Provider 接入 | 梗生命周期管理 |
| 角色卡编辑 Web UI | 三层注入防护架构 |
| — | Viewer 层（per-user 记忆） |

---

## 6. 总结

**AIRI** 和 **AI Live** 虽然都属于"AI 虚拟角色"领域，但面向截然不同的场景：

- **AIRI** 是一个**通用虚拟伴侣平台**，强调跨平台部署、语音双向交互、虚拟形象渲染和社交平台集成。它的 Monorepo 架构和 40+ 共享包体现了"做平台"的野心，游戏 Agent 能力更是拓展了 AI 角色的应用边界。

- **AI Live** 是一个**深度垂直的直播间 AI 系统**，核心差异化在于 VLM 画面理解（AI 真正看懂视频内容）和直播场景的精细建模（弹幕优先级、双轨触发、回复决策、话题管理、情绪好感度）。六层记忆、梗管理、回复验证等设计都服务于"让 AI 主播在直播间表现得自然且有角色深度"这一目标。

**一句话概括**：AIRI 做广度（多平台、多形态、多场景），AI Live 做深度（单一场景下的极致交互质量）。两者在记忆系统和 LLM 分级策略上有相似设计，但在视觉理解、直播交互、情绪建模等维度各有所长，互补空间大。
