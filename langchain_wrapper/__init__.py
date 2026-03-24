"""
langchain_wrapper 模块
提供 LLM 交互层封装
"""

from .model_provider import ModelType, ModelProvider
from .contracts import ContextBlock, RetrievedContextBundle, ModelInvocation
from .pipeline import (
  StreamingPipeline,
  Processor,
  build_system_prompt,
  split_context_channels,
  # 内置处理器
  strip_whitespace,
  remove_empty_lines,
  limit_length,
  replace_text,
  add_prefix,
  add_suffix,
)
from .retriever import RetrieverResolver
from .wrapper import LLMWrapper

__all__ = [
  "ModelType",
  "ModelProvider",
  "ContextBlock",
  "RetrievedContextBundle",
  "ModelInvocation",
  "StreamingPipeline",
  "build_system_prompt",
  "split_context_channels",
  "RetrieverResolver",
  "LLMWrapper",
  # 处理器类型和内置处理器
  "Processor",
  "strip_whitespace",
  "remove_empty_lines",
  "limit_length",
  "replace_text",
  "add_prefix",
  "add_suffix",
]
