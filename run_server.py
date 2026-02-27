"""
启动 WebSocket 服务器
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录添加到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import LLMWrapper, ModelType
from streaming_studio import StreamingStudio
from connection import StreamServiceHost, SpeechBroadcaster


async def main():
  print("=" * 50)
  print("虚拟直播间 WebSocket 服务器")
  print("=" * 50)
  print()

  # 选择人设
  print("请选择主播人设:")
  print("1. karin - 元气偶像少女")
  print("2. sage - 知性学者")
  print("3. kuro - 酷酷游戏主播")
  print()

  while True:
    choice = input("请输入选项 (1/2/3): ").strip()
    if choice == "1":
      persona = "karin"
      break
    elif choice == "2":
      persona = "sage"
      break
    elif choice == "3":
      persona = "kuro"
      break
    else:
      print("无效选项，请重新输入")

  print()

  # 选择模型
  print("请选择模型类型:")
  print("1. OpenAI (需要 API Key)")
  print("2. Anthropic (需要 API Key)")
  print("3. 本地 Qwen (需要本地部署)")
  print()

  while True:
    choice = input("请输入选项 (1/2/3): ").strip()
    if choice == "1":
      model_type = ModelType.OPENAI
      break
    elif choice == "2":
      model_type = ModelType.ANTHROPIC
      break
    elif choice == "3":
      model_type = ModelType.LOCAL_QWEN
      break
    else:
      print("无效选项，请重新输入")

  print()

  # 端口配置
  port_str = input("WebSocket 端口 (默认 8765): ").strip()
  port = int(port_str) if port_str else 8765
  print()

  # 语音服务配置
  speech_url = input("语音服务 URL (留空跳过，如 http://10.81.7.115:9200/say): ").strip()
  print()

  # 初始化
  print("正在初始化...")

  try:
    llm_wrapper = LLMWrapper(model_type=model_type, persona=persona)
    studio = StreamingStudio(llm_wrapper=llm_wrapper)
    host = StreamServiceHost(studio=studio, port=port)

    if speech_url:
      speech = SpeechBroadcaster(api_url=speech_url, model_type=model_type)
      speech.attach(studio)

    # 启动服务
    await studio.start()
    await host.start()

    print()
    print("服务已启动！按 Ctrl+C 停止")
    print()

    # 保持运行
    while True:
      await asyncio.sleep(1)

  except KeyboardInterrupt:
    print("\n正在停止服务...")
  except Exception as e:
    print(f"错误: {e}")
  finally:
    if 'host' in locals():
      await host.stop()
    if 'studio' in locals():
      await studio.stop()


if __name__ == "__main__":
  asyncio.run(main())
