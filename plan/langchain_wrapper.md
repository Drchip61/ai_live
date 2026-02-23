# langchain_wrapper 模块计划

## 模块职责

封装 LangChain 调用，提供可切换的模型源和统一的调用接口。

## 文件结构

```
langchain_wrapper/
├── __init__.py       # 模块导出
├── model_provider.py # ModelType枚举 + ModelProvider类（含预设工厂方法）
├── pipeline.py       # StreamingPipeline (LCEL链 + 前后处理器)
└── wrapper.py        # LLMWrapper简单封装（调用 PromptLoader）
```

## 核心类

### ModelProvider

提供可切换的模型源 + 预设工厂方法。

```python
class ModelType(Enum):
  OPENAI = "openai"
  ANTHROPIC = "anthropic"
  LOCAL_QWEN = "local_qwen"

# 预设远程模型名称映射
REMOTE_MODELS = {
  ModelType.OPENAI: {
    "large": "gpt-5.2",
    "small": "gpt-5-mini",
  },
  ModelType.ANTHROPIC: {
    "large": "claude-opus-4.6",
    "small": "claude-haiku-4.5",
  },
}

class ModelProvider:
  def get_model(model_type, model_name) -> BaseChatModel

  # 预设工厂方法
  @classmethod
  def remote_large(cls, provider=ModelType.OPENAI) -> BaseChatModel
    # OpenAI: gpt-5.2, Anthropic: claude-opus-4.6
  @classmethod
  def remote_small(cls, provider=ModelType.OPENAI) -> BaseChatModel
    # OpenAI: gpt-5-mini, Anthropic: claude-haiku-4.5
  @classmethod
  def local_large(cls) -> BaseChatModel    # Qwen3-8B
  @classmethod
  def local_small(cls) -> BaseChatModel    # Qwen3-1.7B
```

支持的模型源:
- OpenAI API (gpt-5.2, gpt-5-mini)
- Anthropic API (claude-opus-4.6, claude-haiku-4.5)
- 本地Qwen (通过vllm兼容OpenAI接口)

### StreamingPipeline

LangChain LCEL 管道封装。

```python
class StreamingPipeline:
  def __init__(model, system_prompt)
  def invoke(input_text, history) -> str
  async def ainvoke(input_text, history) -> str
  async def astream(input_text, history, extra_context) -> AsyncIterator[str]
```

内部维护两条链：
- `_stream_chain`：基础管道（不含后处理器），用于流式输出
- `_chain`：完整管道（`_stream_chain` + 后处理器），用于非流式调用

```python
# 基础管道（流式使用）
self._stream_chain = (
  RunnableLambda(inject_system_prompt)
  | RunnableLambda(format_history)
  | RunnableLambda(apply_preprocessors)
  | self._prompt_template
  | self.model
  | self._output_parser
)
# 完整管道（非流式使用）
self._chain = self._stream_chain | RunnableLambda(apply_postprocessors)
```

### LLMWrapper

对外简单接口，组合 ModelProvider 和 StreamingPipeline。
通过 PromptLoader 获取系统提示词。

```python
class LLMWrapper:
  def __init__(model_type, model_name, persona)
  def chat(user_input, history) -> str
  async def achat(user_input, history) -> str
  async def achat_stream(user_input, save_history) -> AsyncIterator[str]
```

`achat_stream()` 逐 token yield，流结束后在 `finally` 块中执行后处理 + 历史保存 + 记忆记录（仅在流成功完成时）。

## 状态

- [x] ModelProvider 实现
- [x] 预设模型工厂方法 (remote_large/small, local_large/small)
- [x] StreamingPipeline 实现 (LCEL重构)
- [x] LLMWrapper 实现
- [x] 流式管道 (`_stream_chain` + `astream()`)
- [x] 流式聊天 (`achat_stream()`)
