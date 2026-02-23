"""
弹幕模拟测试工具
每条弹幕自动切换为随机用户身份，一个人模拟直播间多人互动环境
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from coolname import generate

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio, Comment


def random_identity() -> tuple[str, str]:
  """
  生成随机用户身份

  Returns:
    (user_id, nickname) 元组
    - user_id: "adjective_noun" 全小写下划线连接
    - nickname: "Adjective Noun" 首字母大写空格连接
  """
  words = generate(2)
  user_id = "_".join(w.lower() for w in words)
  nickname = " ".join(w.capitalize() for w in words)
  return user_id, nickname


class TestDanmakuStudio:
  """
  弹幕模拟测试类

  每发一条弹幕自动生成一个新的随机用户身份，
  让测试者一个人模拟直播间多人发言的环境。
  """

  def __init__(self):
    self.studio: StreamingStudio = None

  async def setup(self) -> None:
    """初始化设置"""
    print("=" * 50)
    print("弹幕模拟测试工具（每条弹幕 = 随机用户）")
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
    print(f"已选择人设: {persona}")
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

    # 初始化直播间
    print("正在初始化直播间...")
    try:
      self.studio = StreamingStudio(
        persona=persona,
        model_type=model_type,
      )
      print("初始化成功！")
    except Exception as e:
      print(f"初始化失败: {e}")
      raise

    print()
    print("=" * 50)
    print("直播间已开启！每条弹幕将以随机用户身份发送")
    print("输入弹幕后按回车发送")
    print("输入 /quit 退出, /stats 查看统计")
    print("=" * 50)
    print()

  async def run(self) -> None:
    """运行测试"""
    await self.setup()

    await self.studio.start()

    response_task = asyncio.create_task(self._display_responses())

    try:
      await self._input_loop()
    finally:
      response_task.cancel()
      await self.studio.stop()

  async def _input_loop(self) -> None:
    """输入循环，每条弹幕自动切换随机用户"""
    loop = asyncio.get_event_loop()

    while True:
      # 预生成下一个身份，显示在提示符中
      user_id, nickname = random_identity()

      try:
        user_input = await loop.run_in_executor(
          None,
          lambda nick=nickname: input(f"[{nick}] > "),
        )
      except EOFError:
        break

      user_input = user_input.strip()
      if not user_input:
        continue

      if user_input == "/quit":
        print("再见！")
        break
      elif user_input == "/stats":
        stats = self.studio.get_stats()
        print()
        print("统计信息:")
        for key, value in stats.items():
          print(f"  {key}: {value}")
        print()
        continue

      comment = Comment(
        user_id=user_id,
        nickname=nickname,
        content=user_input,
      )
      self.studio.send_comment(comment)

  async def _display_responses(self) -> None:
    """显示主播回复"""
    while True:
      try:
        response = await self.studio.get_response(timeout=1.0)
        if response:
          print()
          print(f"[主播] {response.content}")
          print()
      except asyncio.CancelledError:
        break
      except Exception as e:
        print(f"显示回复错误: {e}")


async def main():
  tester = TestDanmakuStudio()
  try:
    await tester.run()
  except KeyboardInterrupt:
    print("\n已中断")
  except Exception as e:
    print(f"错误: {e}")


if __name__ == "__main__":
  asyncio.run(main())
