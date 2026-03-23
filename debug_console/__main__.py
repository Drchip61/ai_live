"""
python -m debug_console 入口

用法:
  python -m debug_console
  python -m debug_console --port 8080 --persona karin --model openai
"""

import argparse
import logging
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from debug_console import run


MODEL_MAP = {
  "openai": ModelType.OPENAI,
  "anthropic": ModelType.ANTHROPIC,
  "qwen": ModelType.LOCAL_QWEN,
}


def main():
  parser = argparse.ArgumentParser(description="mio-streaming-demo 调试控制台")
  parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认 8080)")
  parser.add_argument("--persona", default="karin", help="角色名称 (karin/sage/kuro/naixiong/dacongming)")
  parser.add_argument(
    "--model", default="openai", choices=list(MODEL_MAP.keys()),
    help="模型类型 (默认 openai)",
  )
  parser.add_argument("--model-name", default=None, help="模型名称 (可选)")
  parser.add_argument(
    "--ephemeral-memory", action="store_true", default=False,
    help="使用临时记忆（进程退出后丢弃，默认持久化到文件）",
  )
  parser.add_argument(
    "--speech-url", default=None,
    help="语音/动作服务 URL（如 http://10.81.7.115:9200/say）",
  )
  parser.add_argument(
    "--controller-url", default=None,
    help="LLM Controller 接口地址（如 http://localhost:2001/v1），指定即启用 Controller 模式",
  )
  parser.add_argument(
    "--controller-model", default="qwen3.5-9b",
    help="Controller 使用的模型名称（默认 qwen3.5-9b）",
  )

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
  )

  try:
    run(
      model_type=MODEL_MAP[args.model],
      model_name=args.model_name,
      persona=args.persona,
      port=args.port,
      enable_global_memory=not args.ephemeral_memory,
      speech_url=args.speech_url,
      controller_url=args.controller_url,
      controller_model=args.controller_model,
    )
  except KeyboardInterrupt:
    print("\n正在关闭...")
    sys.exit(0)


if __name__ == "__main__":
  main()
