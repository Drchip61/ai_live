"""
模型提供者
支持多种模型源的切换
"""

import os
import json
from enum import Enum
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel


class ModelType(Enum):
  """模型类型枚举"""
  OPENAI = "openai"
  ANTHROPIC = "anthropic"
  GEMINI = "gemini"
  LOCAL_QWEN = "local_qwen"


# 预设远程模型名称映射
REMOTE_MODELS = {
  ModelType.OPENAI: {
    "large": "gpt-5.2",
    "small": "gpt-5-mini",
  },
  ModelType.ANTHROPIC: {
    "large": "claude-sonnet-4-20250514",
    "small": "claude-haiku-4-5-20251001",
  },
  ModelType.GEMINI: {
    "large": "gemini-2.5-flash",
    "small": "gemini-2.0-flash",
  },
}


class ModelProvider:
  """
  模型提供者类
  根据模型类型创建相应的 LangChain 模型实例
  """

  def __init__(self, secrets_path: Optional[Path] = None):
    """
    初始化模型提供者

    Args:
      secrets_path: API密钥配置文件路径，默认为项目根目录下的 secrets/api_keys.json
    """
    if secrets_path is None:
      # 默认查找项目根目录下的 secrets 文件夹
      project_root = Path(__file__).parent.parent
      secrets_path = project_root / "secrets" / "api_keys.json"

    self.secrets_path = secrets_path
    self._secrets: dict = {}

    # 尝试加载密钥配置
    if secrets_path.exists():
      self._secrets = json.loads(secrets_path.read_text(encoding="utf-8"))

  def _get_secret(self, key: str) -> Optional[str]:
    """获取密钥，优先从环境变量获取"""
    # 环境变量优先
    env_key = key.upper()
    if env_key in os.environ:
      return os.environ[env_key]
    # 然后从配置文件获取
    return self._secrets.get(key)

  def get_model(
    self,
    model_type: ModelType,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """
    获取指定类型的模型实例

    Args:
      model_type: 模型类型
      model_name: 模型名称，不指定则使用默认值
      **kwargs: 传递给模型的额外参数

    Returns:
      BaseChatModel 实例

    Raises:
      ValueError: 不支持的模型类型或缺少必要配置时抛出
    """
    if model_type == ModelType.OPENAI:
      return self._create_openai_model(model_name, **kwargs)
    elif model_type == ModelType.ANTHROPIC:
      return self._create_anthropic_model(model_name, **kwargs)
    elif model_type == ModelType.GEMINI:
      return self._create_gemini_model(model_name, **kwargs)
    elif model_type == ModelType.LOCAL_QWEN:
      return self._create_local_qwen_model(model_name, **kwargs)
    else:
      raise ValueError(f"不支持的模型类型: {model_type}")

  def _create_openai_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """创建 OpenAI 模型"""
    from langchain_openai import ChatOpenAI

    api_key = self._get_secret("openai_api_key")
    if not api_key:
      raise ValueError("未配置 OpenAI API Key，请设置环境变量 OPENAI_API_KEY 或在 secrets/api_keys.json 中配置")

    return ChatOpenAI(
      model=model_name or "gpt-5.2",
      api_key=api_key,
      **kwargs
    )

  def _create_anthropic_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """创建 Anthropic 模型"""
    from langchain_anthropic import ChatAnthropic

    api_key = self._get_secret("anthropic_api_key")
    if not api_key:
      raise ValueError("未配置 Anthropic API Key，请设置环境变量 ANTHROPIC_API_KEY 或在 secrets/api_keys.json 中配置")

    return ChatAnthropic(
      model=model_name or "claude-sonnet-4-20250514",
      api_key=api_key,
      **kwargs
    )

  def _create_gemini_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """
    创建 Gemini 模型
    通过 Google AI 的 OpenAI 兼容接口调用，无需额外安装包
    """
    from langchain_openai import ChatOpenAI

    api_key = self._get_secret("gemini_api_key")
    if not api_key:
      raise ValueError(
        "未配置 Gemini API Key，请设置环境变量 GEMINI_API_KEY "
        "或在 secrets/api_keys.json 中配置 gemini_api_key\n"
        "获取地址: https://aistudio.google.com/apikey"
      )

    return ChatOpenAI(
      model=model_name or "gemini-2.5-flash",
      api_key=api_key,
      base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
      **kwargs
    )

  def _create_local_qwen_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """
    创建本地 Qwen 模型
    通过 vllm 提供的 OpenAI 兼容接口调用
    """
    from langchain_openai import ChatOpenAI

    base_url = self._get_secret("local_qwen_base_url")
    if not base_url:
      base_url = "http://localhost:8000/v1"

    return ChatOpenAI(
      model=model_name or "Qwen/Qwen3-8B",
      api_key="not-needed",  # 本地部署通常不需要key
      base_url=base_url,
      **kwargs
    )

  # ============================================================
  # 预设模型工厂方法
  # ============================================================

  @classmethod
  def remote_large(
    cls,
    provider: ModelType = ModelType.OPENAI,
    **kwargs
  ) -> BaseChatModel:
    """
    远程大模型

    用途：主对话、复杂推理

    Args:
      provider: 模型源，默认 OpenAI (gpt-5.2)
                支持 ANTHROPIC (claude-opus-4.6)
    """
    model_name = REMOTE_MODELS[provider]["large"]
    return cls().get_model(provider, model_name=model_name, **kwargs)

  @classmethod
  def remote_small(
    cls,
    provider: ModelType = ModelType.OPENAI,
    **kwargs
  ) -> BaseChatModel:
    """
    远程小模型

    用途：支线任务、分类、摘要等轻量计算

    Args:
      provider: 模型源，默认 OpenAI (gpt-5-mini)
                支持 ANTHROPIC (claude-haiku-4.5)
    """
    model_name = REMOTE_MODELS[provider]["small"]
    return cls().get_model(provider, model_name=model_name, **kwargs)

  @classmethod
  def local_large(cls, **kwargs) -> BaseChatModel:
    """
    本地大模型（Qwen3-8B）

    用途：离线主对话、无需 API 的场景
    """
    return cls().get_model(
      ModelType.LOCAL_QWEN,
      model_name="Qwen/Qwen3-8B",
      **kwargs,
    )

  @classmethod
  def local_small(cls, **kwargs) -> BaseChatModel:
    """
    本地小模型（Qwen3-1.7B）

    用途：本地支线任务、资源受限环境
    """
    return cls().get_model(
      ModelType.LOCAL_QWEN,
      model_name="Qwen/Qwen3-1.7B",
      **kwargs,
    )
