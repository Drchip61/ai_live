"""
命令行测试工具
用于测试直播间功能
"""

import asyncio
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio, Comment


class TestChatterStudio:
  """命令行测试类"""

  def __init__(self):
    self.studio: StreamingStudio = None
    self.user_id: str = ""
    self.nickname: str = ""

  async def setup(self) -> None:
    """初始化设置"""
    print("=" * 50)
    print("虚拟直播间测试工具")
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

    # 获取用户信息
    self.user_id = input("请输入你的用户ID: ").strip() or "test_user"
    self.nickname = input("请输入你的昵称: ").strip() or "测试用户"
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
    print("直播间已开启！输入弹幕后按回车发送")
    print("输入 /quit 退出, /stats 查看统计")
    print("=" * 50)
    print()

  async def run(self) -> None:
    """运行测试"""
    await self.setup()

    # 启动直播间
    await self.studio.start()

    # 启动回复显示任务
    response_task = asyncio.create_task(self._display_responses())

    try:
      await self._input_loop()
    finally:
      # 清理
      response_task.cancel()
      await self.studio.stop()

  async def _input_loop(self) -> None:
    """输入循环"""
    loop = asyncio.get_event_loop()

    while True:
      try:
        # 在单独的线程中读取输入
        user_input = await loop.run_in_executor(
          None,
          lambda: input(f"[{self.nickname}] > ")
        )
      except EOFError:
        break

      user_input = user_input.strip()
      if not user_input:
        continue

      # 处理命令
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

      # 发送弹幕
      comment = Comment(
        user_id=self.user_id,
        nickname=self.nickname,
        content=user_input
      )
      self.studio.send_comment(comment)

  async def _display_responses(self) -> None:
    """显示回复的任务"""
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
  tester = TestChatterStudio()
  try:
    await tester.run()
  except KeyboardInterrupt:
    print("\n已中断")
  except Exception as e:
    print(f"错误: {e}")


if __name__ == "__main__":
  asyncio.run(main())
