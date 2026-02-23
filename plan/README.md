# 虚拟直播间 LLM Wrapper - 项目计划

## 项目概述

让 LLM 扮演虚拟主播与观众互动的 wrapper 项目。

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    WebSocket 客户端                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │ 输入客户端│   │ 输入客户端│   │ 输出客户端│ ...            │
│  └────┬─────┘   └────┬─────┘   └────▲─────┘                 │
│       │              │              │                        │
└───────┼──────────────┼──────────────┼────────────────────────┘
        │              │              │
        ▼              ▼              │
┌─────────────────────────────────────┴────────────────────────┐
│                   connection 层                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              StreamServiceHost                       │    │
│  │   - 管理 WebSocket 连接                              │    │
│  │   - 区分 input/output 角色                           │    │
│  │   - 广播回复给输出客户端                             │    │
│  └───────────────────────┬─────────────────────────────┘    │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  streaming_studio 层                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ Comment缓冲区  │  │StreamingStudio │  │CommentDatabase│  │
│  │(deque环形队列) │──│双轨定时器机制  │──│   (SQLite)    │  │
│  └────────────────┘  └───────┬────────┘  └───────────────┘  │
└──────────────────────────────┼───────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
┌──────────────────────────────┐  ┌──────────────────────────┐
│    langchain_wrapper 层      │  │       memory 层          │
│  ┌──────────┐ ┌───────────┐ │  │  ┌────────┐ ┌─────────┐ │
│  │LLMWrapper│─│Pipeline   │ │  │  │ active │ │temporary│ │
│  │  简单接口 │ │ LCEL管道   │ │  │  │ (FIFO) │ │ (RAG)   │ │
│  └──────────┘ └───────────┘ │  │  ├────────┤ ├─────────┤ │
│  ┌──────────────────────┐   │  │  │summary │ │ static  │ │
│  │    ModelProvider      │   │  │  │(定期汇总)│ │(预设记忆)│ │
│  │ 模型源切换+预设工厂    │   │  │  └────────┘ └─────────┘ │
│  └──────────┬───────────┘   │  │  ┌─────────────────────┐ │
└─────────────┼───────────────┘  │  │  MemoryRetriever    │ │
              │                  │  │  跨层检索+LCEL集成    │ │
              ▼                  │  └─────────────────────┘ │
┌──────────────────────────────┐ └──────────┬───────────────┘
│  ┌──────────┐  ┌──────────┐ │             │
│  │  OpenAI  │  │ 本地Qwen │ │             ▼
│  │   API    │  │  (vllm)  │ │  ┌─────────────────────────┐
│  └──────────┘  └──────────┘ │  │      personas/          │
└──────────────────────────────┘  │  角色数据 + static_memories │
              ▲                   └─────────────────────────┘
              │
┌─────────────┴────────────────┐
│         prompts/             │
│  base_instruction + personas │
│     (PromptLoader)           │
└──────────────────────────────┘
```

## 模块说明

| 模块 | 职责 | 计划文件 |
|------|------|----------|
| prompts | 通用提示词加载，委托 PersonaLoader | [prompts.md](prompts.md) |
| personas | 角色人格管理（prompt + 预设记忆） | [personas.md](personas.md) |
| langchain_wrapper | LLM 调用封装 | [langchain_wrapper.md](langchain_wrapper.md) |
| memory | 分层记忆系统（RAG） | [memory.md](memory.md) |
| streaming_studio | 直播间核心逻辑 | [streaming_studio.md](streaming_studio.md) |
| connection | WebSocket 服务 | [connection.md](connection.md) |
| debug_console | NiceGUI 调试控制台 | [debug_console.md](debug_console.md) |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

创建 `secrets/api_keys.json`:

```json
{
  "openai_api_key": "your-key-here",
  "anthropic_api_key": "your-key-here"
}
```

或设置环境变量:

```bash
export OPENAI_API_KEY="your-key-here"
```

### 3. 命令行测试

```bash
python -m streaming_studio.test_chatter_studio
```

### 4. WebSocket 服务

启动服务器:
```bash
python run_server.py
```

连接客户端:
```bash
python -m connection.test_chatter_web
```

## 完成状态

- [x] Phase 1: 基础层 (secrets, prompts)
- [x] Phase 2: LLM 交互层 (langchain_wrapper)
- [x] Phase 3: 直播间层 (streaming_studio)
- [x] Phase 4: WebSocket 层 (connection)
- [x] Phase 5: 计划文档
- [x] Phase 6: 角色人格拆分 (personas/)
- [x] Phase 7: 预设模型工厂方法 (ModelProvider)
- [x] Phase 8: 分层记忆系统 (memory/)
- [x] Phase 9: 双轨制回复触发机制 (streaming_studio/)
- [x] Phase 10: NiceGUI 调试控制台 (debug_console/)
- [x] Phase 11: 全链路流式回复 + 模拟直播间左右分栏
- [x] Phase 12: 监控面板排版优化 + 冷场应对策略 + 角色记忆扩充
