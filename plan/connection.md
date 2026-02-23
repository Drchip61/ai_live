# connection 模块计划

## 模块职责

WebSocket 服务层，支持多客户端连接，区分输入者和输出者角色。

## 文件结构

```
connection/
├── __init__.py             # 模块导出
├── stream_service_host.py  # WebSocket服务
└── test_chatter_web.py     # WebSocket测试客户端
```

## 协议设计

### 连接握手

客户端连接后发送角色声明:

```json
{"type": "register", "role": "input"}   // 弹幕输入者
{"type": "register", "role": "output"}  // 回复接收者
```

### 消息格式

弹幕消息 (input -> server):
```json
{
  "type": "comment",
  "user_id": "user_001",
  "nickname": "小明",
  "content": "主播好！"
}
```

回复消息 (server -> output):
```json
{
  "type": "response",
  "content": "小明好呀~欢迎来到直播间！",
  "reply_to": ["comment_001"]
}
```

流式回复片段 (server -> output, 当 studio 启用流式时):
```json
{
  "type": "response_chunk",
  "response_id": "uuid-xxx",
  "chunk": "小明",
  "accumulated": "小明",
  "done": false
}
```

流式完成后仍会发送完整的 `response` 消息，客户端可自行选择使用哪种。

## 核心类

### StreamServiceHost

```python
class StreamServiceHost:
  def __init__(studio, host, port)
  async def start()
  async def stop()
  async def broadcast_response(response)
```

## 测试客户端

test_chatter_web.py 提供交互式命令行客户端:
- 连接WebSocket服务
- 输入用户信息
- 发送弹幕
- 接收并显示回复

## 状态

- [x] StreamServiceHost实现
- [x] 测试客户端实现
- [x] 流式回复片段广播 (`response_chunk`)
