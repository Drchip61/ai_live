"""
ModelProvider 回归测试
"""

import sys
import types
from unittest.mock import MagicMock, patch

from langchain_wrapper.model_provider import ModelProvider


def _capture_local_qwen_kwargs(provider: ModelProvider, **kwargs):
  fake_ctor = MagicMock(return_value="fake-chat-model")
  fake_module = types.SimpleNamespace(ChatOpenAI=fake_ctor)
  with patch.dict(sys.modules, {"langchain_openai": fake_module}):
    model = provider._create_local_qwen_model(**kwargs)
  assert model == "fake-chat-model"
  return fake_ctor.call_args.kwargs


def test_local_qwen_explicit_ollama_injects_keep_alive():
  provider = ModelProvider()
  kwargs = _capture_local_qwen_kwargs(
    provider,
    model_name="qwen-test",
    base_url="http://proxy.local/v1",
    is_ollama=True,
  )
  assert kwargs["extra_body"]["keep_alive"] == -1
  print("  [PASS] 显式 is_ollama=True 会注入 keep_alive=-1")


def test_local_qwen_ollama_secret_override_supports_duration_string():
  provider = ModelProvider()
  provider._secrets = {
    "local_qwen_is_ollama": True,
    "ollama_keep_alive": "30m",
  }
  kwargs = _capture_local_qwen_kwargs(
    provider,
    model_name="qwen-test",
    base_url="http://proxy.local/v1",
    extra_body={"seed": 7},
  )
  assert kwargs["extra_body"]["seed"] == 7
  assert kwargs["extra_body"]["keep_alive"] == "30m"
  print("  [PASS] secrets 可强制 Ollama 并透传 keep_alive 时长")


def test_local_qwen_explicit_disable_skips_keep_alive_injection():
  provider = ModelProvider()
  kwargs = _capture_local_qwen_kwargs(
    provider,
    model_name="qwen-test",
    base_url="http://localhost:11434/v1",
    is_ollama=False,
  )
  assert "extra_body" not in kwargs or "keep_alive" not in kwargs["extra_body"]
  print("  [PASS] 显式 is_ollama=False 会关闭 keep_alive 注入")


def main():
  tests = [
    test_local_qwen_explicit_ollama_injects_keep_alive,
    test_local_qwen_ollama_secret_override_supports_duration_string,
    test_local_qwen_explicit_disable_skips_keep_alive_injection,
  ]

  failed = 0
  for test_fn in tests:
    try:
      test_fn()
    except AssertionError as exc:
      failed += 1
      print(f"  [FAIL] {test_fn.__name__}: {exc}")
    except Exception as exc:
      failed += 1
      print(f"  [ERROR] {test_fn.__name__}: {type(exc).__name__}: {exc}")

  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
