# streaming_studio 模块计划

## 模块职责

虚拟直播间核心，管理弹幕缓冲、双轨定时器、LLM调用和回复分发。

## 文件结构

```
streaming_studio/
├── __init__.py               # 模块导出
├── models.py                 # Comment, StreamerResponse 数据模型
├── database.py               # SQLite弹幕存储
├── config.py                 # StudioConfig 行为配置
├── studio.py                 # StreamingStudio 异步核心类
├── test_chatter_studio.py    # 命令行测试（单用户）
└── test_danmaku_studio.py    # 弹幕模拟测试（随机用户身份）
```

## 数据模型

### Comment (弹幕)

```python
@dataclass
class Comment:
  id: str              # 唯一ID
  user_id: str         # 用户ID
  nickname: str        # 用户昵称
  content: str         # 弹幕内容
  timestamp: datetime  # 发送时间
```

### StreamerResponse (主播回复)

```python
@dataclass
class StreamerResponse:
  id: str              # 唯一ID
  content: str         # 回复内容
  reply_to: list[str]  # 回复的弹幕ID列表
  timestamp: datetime  # 回复时间
```

### ResponseChunk (流式回复片段)

```python
@dataclass(frozen=True)
class ResponseChunk:
  response_id: str      # 所属回复的 ID
  chunk: str            # 本次新增的文本片段
  accumulated: str      # 截至目前的累积文本
  done: bool = False    # 是否为最后一个片段
```

## 核心类

### StudioConfig

行为配置，管理回复节奏、缓冲区大小等细节参数。

```python
@dataclass(frozen=True)
class StudioConfig:
  min_interval: float = 1.0         # 回复最小间隔（秒）
  max_interval: float = 10.0        # 回复最大间隔（秒）
  recent_comments_limit: int = 20   # 每次回复收集的最近弹幕数
  buffer_maxlen: int = 200          # 弹幕缓冲区最大容量
```

### CommentDatabase

SQLite存储，保存弹幕和回复记录。

```python
class CommentDatabase:
  def __init__(db_path)
  def save_comment(comment)
  def save_response(response)
  def get_recent_comments(limit) -> list[Comment]
```

### StreamingStudio

异步主循环，双轨定时器驱动的回复触发机制。

#### 构造参数

**核心配置：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `persona` | str | "karin" | 主播人设 (karin/sage/kuro) |
| `model_type` | ModelType | OPENAI | 模型类型 (OPENAI/ANTHROPIC/LOCAL_QWEN) |
| `model_name` | str \| None | None | 模型名称（可选） |
| `enable_memory` | bool | False | 是否启用分层记忆系统 |

**高级定制：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_wrapper` | LLMWrapper \| None | None | 自定义 LLM 封装（传入后忽略上述核心配置） |
| `database` | CommentDatabase \| None | None | 自定义数据库 |
| `config` | StudioConfig \| None | None | 自定义行为配置 |

#### 双轨定时器机制

核心思想：**定时器 + 弹幕加速**

```
每次回复间隔 = random(min_interval, max_interval)
当有新弹幕到达时：剩余等待时间 -= 弹幕数量
```

实现细节：
- 使用 `deque[Comment]` (maxlen=200) 作为环形缓冲区
- 使用 `asyncio.Event` 通知新弹幕到达
- 根据 `_last_reply_time` 将弹幕分为"旧弹幕"和"新弹幕"
- 格式化时标注相对时间（如"35秒前"）和绝对时间（如"14:23:05"）
- 检测沉默状态，当无新弹幕时计算并显示距上次弹幕的秒数

#### 流式回复

运行时属性 `enable_streaming: bool`（默认 `False`），由上游调用方在运行时设置：

```python
studio = StreamingStudio(persona="karin", model_type=ModelType.OPENAI)
studio.enable_streaming = True  # 运行时切换，非构造参数
```

启用后，`_main_loop` 调用 `_generate_response_streaming()` 代替 `_generate_response()`，
通过 `llm_wrapper.achat_stream()` 逐 token 生成并分发 `ResponseChunk`。
完成后仍返回完整的 `StreamerResponse` 走正常的保存/回调流程。

#### 方法列表

```python
class StreamingStudio:
  def __init__(llm_wrapper, database, recent_comments_limit, min_interval, max_interval)

  # 生命周期
  async def start()
  async def stop()

  # 弹幕接收
  def send_comment(comment: Comment)

  # 回复获取
  async def get_response(timeout=None) -> StreamerResponse

  # 回调机制（完整回复）
  def on_response(callback: Callable[[StreamerResponse], None])
  def remove_callback(callback)

  # 回调机制（流式片段）
  def on_response_chunk(callback: Callable[[ResponseChunk], None])
  def remove_chunk_callback(callback)

  # 调试监控
  def debug_state() -> dict
  def get_stats() -> dict
```

#### 弹幕格式化

单条弹幕格式：
```
[14:23:05 / 35秒前] 花凛 (id: user_abc): 主播唱首歌
```

组合 prompt 格式：
```
【上次回复前的弹幕（背景参考）】
- [14:22:00 / 3分35秒前] 小明 (id: u001): 你好
- [14:22:30 / 3分5秒前] 小红 (id: u002): 刚来

【上次回复后的新弹幕】
- [14:25:00 / 35秒前] 小明 (id: u001): 主播唱歌
- [14:25:20 / 15秒前] 小蓝 (id: u003): 好耶
```

沉默检测：
```
【上次回复后无人说话】
（已经 45 秒没人说话了）
```

## 使用示例

### 基本用法（推荐）

```python
from streaming_studio import StreamingStudio, Comment
from langchain_wrapper import ModelType

# 创建直播间
studio = StreamingStudio(
    persona="karin",
    model_type=ModelType.OPENAI,
    enable_memory=False,  # 默认不启用记忆
)

# 启动
await studio.start()

# 发送弹幕
studio.send_comment(Comment(
    user_id="user_001",
    nickname="小明",
    content="主播好！",
))

# 获取回复
response = await studio.get_response()
print(response.content)

# 停止
await studio.stop()
```

### 启用记忆系统

```python
studio = StreamingStudio(
    persona="karin",
    model_type=ModelType.OPENAI,
    enable_memory=True,  # 启用分层记忆
)
```

### 自定义行为配置

```python
from streaming_studio import StreamingStudio, StudioConfig

studio = StreamingStudio(
    persona="karin",
    config=StudioConfig(
        min_interval=2.0,           # 回复间隔 2-15 秒
        max_interval=15.0,
        recent_comments_limit=30,   # 每次考虑最近 30 条弹幕
        buffer_maxlen=500,          # 缓冲区容量 500
    ),
)
```

### 高级定制（自定义 LLMWrapper）

```python
from langchain_wrapper import LLMWrapper, ModelType

# 手动创建 LLMWrapper
llm = LLMWrapper(
    model_type=ModelType.ANTHROPIC,
    model_name="claude-opus-4.6",
    persona="sage",
    max_history=50,  # 保留更多历史
)

# 传入自定义 wrapper
studio = StreamingStudio(llm_wrapper=llm)
```

## 核心特性

### 双轨定时器

模拟真人主播"隔一会儿说一句，有人刷屏就加快节奏"的自然行为：
- 基础节奏：每次回复后等待 random(1s, 10s)
- 加速机制：每条新弹幕到达时减少剩余等待 1 秒
- 避免被动触发：不再完全依赖"收到弹幕才回复"

### 时间观念

- 每条弹幕标注绝对时间 + 相对时间
- 区分"旧弹幕（背景）"和"新弹幕（需回复）"
- 检测沉默状态，提示 AI 主动活跃气氛
- base_instruction.txt 中增加时间观念指导

### 弹幕缓冲区

- 使用 deque(maxlen=200) 保留历史上下文
- 支持回看最近弹幕，提供更丰富的背景信息
- 每次触发时取最近 20 条弹幕

## 测试工具

### test_chatter_studio.py
单用户命令行测试，手动输入弹幕。

### test_danmaku_studio.py
多用户弹幕模拟测试：
- 使用 coolname 库生成随机用户身份
- 每条弹幕自动分配不同用户 ID 和昵称
- 验证 AI 对多用户场景的适应能力

## 状态

- [x] 数据模型定义
- [x] 数据库实现
- [x] StudioConfig 行为配置
- [x] StreamingStudio 核心实现
- [x] 双轨定时器机制
- [x] 弹幕格式化（时间标注）
- [x] 沉默检测
- [x] 简化 API（persona/model_type/enable_memory 直接传入）
- [x] debug_state() 调试接口
- [x] 命令行测试（单用户）
- [x] 弹幕模拟测试（随机用户）
- [x] ResponseChunk 数据模型
- [x] 流式回复生成 (`_generate_response_streaming`)
- [x] 流式片段回调 (`on_response_chunk` / `remove_chunk_callback`)
