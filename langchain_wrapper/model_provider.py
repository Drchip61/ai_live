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
  DEEPSEEK = "deepseek"
  QWEN = "qwen"
  LOCAL_QWEN = "local_qwen"


# OpenAI 默认大模型
DEFAULT_OPENAI_LARGE = "gpt-5.4"

# Anthropic 默认大模型（统一单一来源，避免多处硬编码不一致）
DEFAULT_ANTHROPIC_LARGE = "claude-sonnet-4-6"
# Anthropic 默认小模型（使用无日期别名，便于平滑升级）
DEFAULT_ANTHROPIC_SMALL = "claude-haiku-4-5"
# DeepSeek 当前低时延非推理模型
DEFAULT_DEEPSEEK_FAST = "deepseek-chat"
# Qwen 远程小模型默认使用低时延商业版
DEFAULT_QWEN_FAST = "qwen3.5-flash"


# 预设远程模型名称映射
REMOTE_MODELS = {
  ModelType.OPENAI: {
    "large": DEFAULT_OPENAI_LARGE,
    "small": "gpt-5-mini",
  },
  ModelType.ANTHROPIC: {
    "large": DEFAULT_ANTHROPIC_LARGE,
    "small": DEFAULT_ANTHROPIC_SMALL,
  },
  ModelType.GEMINI: {
    "large": "gemini-3-flash",
    "small": "gemini-2.5-flash-lite",
  },
  # 目前统一用 deepseek-chat，避免误切到更慢的 reasoning 模式。
  ModelType.DEEPSEEK: {
    "large": DEFAULT_DEEPSEEK_FAST,
    "small": DEFAULT_DEEPSEEK_FAST,
  },
  ModelType.QWEN: {
    "large": "qwen3.5-plus",
    "small": DEFAULT_QWEN_FAST,
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

  def _get_first_secret(self, *keys: str) -> Optional[str]:
    """按顺序读取第一个可用密钥/配置。"""
    for key in keys:
      value = self._get_secret(key)
      if value:
        return value
    return None

  @staticmethod
  def _parse_bool_flag(value) -> Optional[bool]:
    """将环境变量/配置中的真假值规范化。"""
    if value is None:
      return None
    if isinstance(value, bool):
      return value
    if isinstance(value, (int, float)):
      return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
      return True
    if text in {"0", "false", "no", "off", "n"}:
      return False
    return None

  @staticmethod
  def _parse_keep_alive_value(value):
    """支持 Ollama keep_alive 的整数秒数或时长字符串。"""
    if value in (None, ""):
      return -1
    if isinstance(value, (int, float)):
      return int(value)

    text = str(value).strip()
    if text.lstrip("-").isdigit():
      return int(text)
    return text

  def _looks_like_ollama(
    self,
    base_url: Optional[str],
    *,
    explicit_flag=None,
  ) -> bool:
    """判断 local_qwen 实际是否走 Ollama 兼容接口。"""
    explicit_value = self._parse_bool_flag(explicit_flag)
    if explicit_value is not None:
      return explicit_value

    configured_value = self._parse_bool_flag(
      self._get_first_secret("local_qwen_is_ollama")
    )
    if configured_value is not None:
      return configured_value

    base = (base_url or "").strip().lower()
    if not base:
      return False
    return (
      ":11434" in base
      or "://ollama" in base
      or ".ollama" in base
      or "/ollama" in base
      or "/api/chat" in base
      or "/api/generate" in base
    )

  def _resolve_ollama_keep_alive(self, explicit_value=None):
    """读取 keep_alive，默认 -1 表示常驻不卸载。"""
    if explicit_value not in (None, ""):
      return self._parse_keep_alive_value(explicit_value)

    configured_value = self._get_first_secret(
      "local_qwen_keep_alive",
      "ollama_keep_alive",
    )
    return self._parse_keep_alive_value(configured_value)

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
    elif model_type == ModelType.DEEPSEEK:
      return self._create_deepseek_model(model_name, **kwargs)
    elif model_type == ModelType.QWEN:
      return self._create_qwen_model(model_name, **kwargs)
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
      model=model_name or DEFAULT_OPENAI_LARGE,
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
      model=model_name or DEFAULT_ANTHROPIC_LARGE,
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

  def _create_deepseek_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """创建 DeepSeek 模型（OpenAI 兼容接口）。"""
    from langchain_openai import ChatOpenAI

    api_key = kwargs.pop("api_key", None)
    if not api_key:
      api_key = self._get_secret("deepseek_api_key")
    if not api_key:
      raise ValueError(
        "未配置 DeepSeek API Key，请设置环境变量 DEEPSEEK_API_KEY "
        "或在 secrets/api_keys.json 中配置 deepseek_api_key"
      )

    base_url = kwargs.pop("base_url", None)
    if not base_url:
      base_url = self._get_secret("deepseek_base_url")
    if not base_url:
      base_url = "https://api.deepseek.com/v1"

    return ChatOpenAI(
      model=model_name or DEFAULT_DEEPSEEK_FAST,
      api_key=api_key,
      base_url=base_url,
      **kwargs
    )

  def _create_qwen_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """创建 Qwen 远程模型（DashScope OpenAI 兼容接口）。"""
    from langchain_openai import ChatOpenAI

    api_key = kwargs.pop("api_key", None)
    if not api_key:
      api_key = self._get_first_secret("qwen_api_key", "dashscope_api_key")
    if not api_key:
      raise ValueError(
        "未配置 Qwen API Key，请设置环境变量 QWEN_API_KEY / DASHSCOPE_API_KEY "
        "或在 secrets/api_keys.json 中配置 qwen_api_key"
      )

    base_url = kwargs.pop("base_url", None)
    if not base_url:
      base_url = self._get_first_secret("qwen_base_url", "dashscope_base_url")
    if not base_url:
      # 默认走国际兼容节点；当前 qwen3.5-flash key 已确认在该节点可识别。
      base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    return ChatOpenAI(
      model=model_name or DEFAULT_QWEN_FAST,
      api_key=api_key,
      base_url=base_url,
      **kwargs
    )

  def _create_local_qwen_model(
    self,
    model_name: Optional[str] = None,
    **kwargs
  ) -> BaseChatModel:
    """
    创建本地 Qwen 模型
    通过本地 OpenAI 兼容接口调用（vLLM / Ollama 等）
    """
    from langchain_openai import ChatOpenAI

    base_url = kwargs.pop("base_url", None)
    if not base_url:
      base_url = self._get_secret("local_qwen_base_url")
    if not base_url:
      base_url = "http://localhost:8000/v1"

    is_ollama = kwargs.pop("is_ollama", None)
    ollama_keep_alive = kwargs.pop("ollama_keep_alive", None)
    extra = kwargs.pop("model_kwargs", None) or {}
    extra_body = kwargs.pop("extra_body", None)
    if extra_body is None:
      extra_body = extra.pop("extra_body", None)

    # Ollama 兼容接口必须把 keep_alive 放进请求体。
    # 仅靠 model_kwargs 不会透传到最终 HTTP body。
    if self._looks_like_ollama(base_url, explicit_flag=is_ollama):
      eb = dict(extra_body or {})
      eb.setdefault("keep_alive", self._resolve_ollama_keep_alive(ollama_keep_alive))
      extra_body = eb

    chat_kwargs = {
      "model": model_name or "Qwen/Qwen3-8B",
      "api_key": "not-needed",
      "base_url": base_url,
      **kwargs,
    }
    if extra_body is not None:
      chat_kwargs["extra_body"] = extra_body
    if extra:
      chat_kwargs["model_kwargs"] = extra

    return ChatOpenAI(**chat_kwargs)

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
      provider: 模型源，默认 OpenAI (gpt-5.4)
                支持 ANTHROPIC (claude-sonnet-4-6) / DEEPSEEK (deepseek-chat)
                / QWEN (qwen3.5-plus)
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
                支持 ANTHROPIC (claude-haiku-4.5) / DEEPSEEK (deepseek-chat)
                / QWEN (qwen3.5-flash)
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

  @classmethod
  def controller(
    cls,
    base_url: str = "http://localhost:2001/v1",
    model_name: str = "qwen3.5-9b",
    **kwargs,
  ) -> BaseChatModel:
    """
    Controller 调度器模型（本地 Qwen 3.5-9B）

    用途：LLM Controller 统一场景化调度
    """
    return cls().get_model(
      ModelType.LOCAL_QWEN,
      model_name=model_name,
      base_url=base_url,
      temperature=0.3,
      max_tokens=512,
      **kwargs,
    )
