# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

mio-streaming-demo 是一个虚拟直播间 LLM wrapper 项目，让 LLM 扮演虚拟主播与观众互动。使用 Python + LangChain 构建。

## 开发命令

```bash
# 激活虚拟环境
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 测试命令行聊天 (待实现)
python -m streaming_studio.test_chatter_studio

# 测试 WebSocket 服务 (待实现)
python -m connection.test_chatter_web

# 弹幕模拟测试（随机用户身份）
python -m streaming_studio.test_danmaku_studio

# NiceGUI 调试控制台（监控面板 + 模拟直播间）
python -m debug_console
python -m debug_console --port 8080 --persona karin --model openai

# VLM 模式：视频 + 弹幕 → Claude 多模态直播间
python run_vlm_demo.py --video data/sample.mp4 --danmaku data/sample.xml
python run_vlm_demo.py --video data/sample.mp4 --danmaku data/sample.xml --persona sage --speed 2.0
```

## 架构设计

```
langchain_wrapper/    # LLM 交互层：模型源切换、LCEL管道、对外wrapper（支持多模态图片输入）
  model_provider.py   #   ModelType枚举 + ModelProvider（含预设工厂方法）
  pipeline.py         #   StreamingPipeline（LCEL链 + 多模态路径）
  wrapper.py          #   LLMWrapper 简单封装（调用 PromptLoader 获取 system prompt）

video_source/         # 视频源模块（VLM 模式）
  frame_extractor.py  #   OpenCV 帧提取 + base64 JPEG 编码
  danmaku_parser.py   #   B站弹幕 XML 解析器
  video_player.py     #   异步时间轴播放器（帧+弹幕同步产出）

memory/               # 分层记忆系统（独立顶层模块）
  config.py           #   全局配置（ActiveConfig/TemporaryConfig/SummaryConfig 等）
  significance.py     #   significance 评分函数（decay/boost/initial）
  store.py            #   VectorStore（Chroma 封装）
  archive.py          #   MemoryArchive（归档到 JSON）
  retriever.py        #   MemoryRetriever 跨层检索 + LCEL Runnable
  formatter.py        #   记忆格式化（相对时间显示）
  layers/             #   四层记忆实现
    base.py           #     MemoryEntry 数据类
    active.py         #     ActiveLayer（FIFO，溢出回调）
    temporary.py      #     TemporaryLayer（RAG + significance 衰减）
    summary.py        #     SummaryLayer（定期汇总 + 清理）
    static.py         #     StaticLayer（预设记忆，从 JSON 加载）

personas/             # 角色人格管理
  persona_loader.py   #   PersonaLoader（自动发现角色子目录）
  {角色名}/           #   每个角色一个子目录
    system_prompt.txt  #     角色专属系统提示词
    static_memories/   #     预设记忆 JSON 文件

prompts/              # 通用提示词
  prompt_loader.py    #   PromptLoader（加载 base_instruction + 委托 PersonaLoader）
  base_instruction.txt #  主播基础指令

topic_manager/        # 话题管理器（可选模块，enable_topic_manager 开启）
  config.py           #   TopicManagerConfig（所有可调参数）
  models.py           #   Topic 数据类（frozen dataclass）
  table.py            #   TopicTable（内存话题表，CRUD + 衰减 + 清理）
  classifier.py       #   弹幕分类器（规则匹配降级 → 小模型；单条/批量模式）
  analyzer.py         #   回复后分析器（内容分析 + 节奏分析，2 个并行异步任务）
  formatter.py        #   话题输出格式化（弹幕标注 + 话题摘要 + 额外指令）
  prompts.py          #   所有 LLM prompt 模板
  manager.py          #   TopicManager 编排器（生命周期 + 弹幕转发 + 分析调度）

streaming_studio/     # 虚拟直播间：异步运行、弹幕缓冲、双轨定时器、回复分发、SQLite存储、VLM视频集成
connection/           # WebSocket层：StreamServiceHost、多客户端订阅

debug_console/        # NiceGUI 本地调试控制台
  app.py              #   应用入口 + 顶级菜单路由
  state_collector.py  #   聚合各模块 debug_state() 的状态收集器
  pages/
    monitor.py        #   监控面板（实时显示记忆/弹幕/prompt/定时器状态）
    chat.py           #   模拟直播间（单用户/多用户随机身份模式）
secrets/              # API密钥等敏感信息(gitignore)
plan/                 # 工作计划文档
spec/                 # 项目规范(只读，不要修改)
```

### 依赖方向

```
connection → streaming_studio → langchain_wrapper（支持 images 多模态参数）
                              → memory
                              → video_source（VLM 模式：帧提取 + 弹幕注入）
                                  ↓
                              personas/（读取 static_memories）
prompts → personas/（读取 system_prompt）
langchain_wrapper/wrapper → prompts/（获取完整 system prompt）
debug_console → streaming_studio（读取 debug_state）
              → langchain_wrapper（读取 debug_state）
              → memory（读取 debug_state）
              → topic_manager（读取 debug_state，通过 streaming_studio 间接）
streaming_studio → topic_manager（弹幕转发 + 上下文获取 + 回复后分析）
```

## 可用角色

- **karin** — 元气偶像少女
- **sage** — 知性学者
- **kuro** — 酷酷游戏主播

## 编码规范

- 命名：snake_case
- 缩进：2空格
- 注释/文档：中文
- 所有路径以项目根目录为基准
- 数据类使用 `@dataclass(frozen=True)` 保持不可变

## 当前阶段

草稿阶段：搭建框架，代码简洁可读，不考虑生产环境部署需求。
