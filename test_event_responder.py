"""
事件回复系统端到端测试
模拟各种弹幕、入场、礼物、上舰事件，验证三级调度逻辑
"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper import ModelType
from streaming_studio import StreamingStudio, Comment, EventType


async def main():
  print("=" * 60)
  print("  事件回复系统端到端测试")
  print("=" * 60)
  print()

  studio = StreamingStudio(
    persona="mio",
    model_type=ModelType.ANTHROPIC,
    enable_memory=False,
    enable_global_memory=False,
    enable_topic_manager=False,
    enable_reply_decider=False,
    enable_comment_clusterer=False,
  )

  responses: list[str] = []

  def on_resp(resp):
    tag = f"[{resp.response_style}]"
    print(f"  >>> {tag} {resp.content[:80]}")
    responses.append(f"{tag} {resp.content}")

  studio.on_response(on_resp)
  await studio.start()

  # 等启动
  await asyncio.sleep(1)

  # ──────────────────────────────────────────────
  print("\n── 测试 1: 普通入场（低优先，前缀模式）──")
  print("   发送 3 条入场事件，不发弹幕，验证不会独立回复")
  for i, name in enumerate(["小明", "小红", "小蓝"]):
    studio.send_comment(Comment(
      user_id=f"entry_{i}", nickname=name, content="",
      event_type=EventType.ENTRY,
    ))
  print(f"   队列状态: {studio._event_responder.debug_state()}")
  await asyncio.sleep(2)
  print(f"   等待 2 秒后回复数: {len(responses)} (应该为 0，入场不独立回复)")

  # ──────────────────────────────────────────────
  print("\n── 测试 2: 普通弹幕触发前缀模式 ──")
  print("   发送一条普通弹幕，验证前缀问候 + LLM 回复")
  before = len(responses)
  studio.send_comment(Comment(
    user_id="viewer_1", nickname="观众A", content="主播好！",
    event_type=EventType.DANMAKU,
  ))
  await asyncio.sleep(12)
  after = len(responses)
  new_count = after - before
  print(f"   新增回复: {new_count} 条")
  has_prefix = any("[prefix_greet]" in r for r in responses[before:])
  print(f"   包含前缀问候: {'是' if has_prefix else '否'}")

  # ──────────────────────────────────────────────
  print("\n── 测试 3: 免费礼物（前缀模式）──")
  print("   发送免费礼物(辣条)，验证归入前缀")
  studio.send_comment(Comment(
    user_id="gift_1", nickname="土豪A", content="",
    event_type=EventType.GIFT, gift_name="辣条", gift_num=5, price=0.0,
  ))
  print(f"   队列状态: {studio._event_responder.debug_state()}")

  # ──────────────────────────────────────────────
  print("\n── 测试 4: 小额礼物（前缀模式，<10元）──")
  studio.send_comment(Comment(
    user_id="gift_2", nickname="土豪B", content="",
    event_type=EventType.GIFT, gift_name="小心心", gift_num=1, price=1.0,
  ))
  print(f"   cheap_gift_pending: {studio._event_responder.debug_state()['cheap_gift_pending']}")

  # ──────────────────────────────────────────────
  print("\n── 测试 5: 中等礼物（高优先，独立回复，>=10元）──")
  before = len(responses)
  studio.send_comment(Comment(
    user_id="gift_3", nickname="土豪C", content="",
    event_type=EventType.GIFT, gift_name="花球", gift_num=1, price=50.0,
  ))
  print(f"   gift_pending (高优先): {studio._event_responder.debug_state()['gift_pending']}")
  await asyncio.sleep(3)
  after = len(responses)
  has_gift_thanks = any("[gift_thanks]" in r for r in responses[before:])
  print(f"   独立礼物感谢: {'是' if has_gift_thanks else '否'}")

  # ──────────────────────────────────────────────
  print("\n── 测试 6: 大额礼物（高优先，独立回复，>=100元）──")
  before = len(responses)
  studio.send_comment(Comment(
    user_id="gift_4", nickname="超级土豪", content="",
    event_type=EventType.GIFT, gift_name="小电视飞船", gift_num=1, price=1245.0,
  ))
  await asyncio.sleep(3)
  after = len(responses)
  has_gift_thanks = any("[gift_thanks]" in r for r in responses[before:])
  print(f"   独立礼物感谢: {'是' if has_gift_thanks else '否'}")

  # ──────────────────────────────────────────────
  print("\n── 测试 7: 上舰事件（最高优先，独立回复）──")
  before = len(responses)
  studio.send_comment(Comment(
    user_id="guard_1", nickname="舰长大人", content="",
    event_type=EventType.GUARD_BUY, guard_level=1, gift_num=1,
  ))
  await asyncio.sleep(3)
  after = len(responses)
  has_guard = any("[guard_thanks]" in r for r in responses[before:])
  print(f"   上舰感谢: {'是' if has_guard else '否'}")

  # ──────────────────────────────────────────────
  print("\n── 测试 8: VIP 入场（舰长进直播间，高优先）──")
  # 先注册舰长身份
  studio._guard_roster.add_or_extend(
    uid="vip_user_1", nickname="VIP舰长", guard_level=1, num_months=1,
  )
  before = len(responses)
  studio.send_comment(Comment(
    user_id="vip_user_1", nickname="VIP舰长", content="",
    event_type=EventType.ENTRY,
  ))
  print(f"   vip_entry_pending: {studio._event_responder.debug_state()['vip_entry_pending']}")
  await asyncio.sleep(3)
  after = len(responses)
  has_vip = any("[vip_entry]" in r for r in responses[before:])
  print(f"   VIP入场欢迎: {'是' if has_vip else '否'}")

  # ──────────────────────────────────────────────
  print("\n── 测试 9: Super Chat（走 LLM 管线）──")
  before = len(responses)
  studio.send_comment(Comment(
    user_id="sc_1", nickname="SC观众", content="主播唱首歌吧！",
    event_type=EventType.SUPER_CHAT, price=30.0,
  ))
  await asyncio.sleep(12)
  after = len(responses)
  print(f"   SC 回复: {after - before} 条 (应该有 LLM 回复)")

  # ──────────────────────────────────────────────
  print("\n── 测试 10: 批量入场 + 弹幕，验证前缀冷却 ──")
  # 重置前缀冷却
  studio._event_responder._last_prefix_time = 0.0
  for i in range(8):
    studio.send_comment(Comment(
      user_id=f"batch_{i}", nickname=f"路人{i+1}", content="",
      event_type=EventType.ENTRY,
    ))
  before = len(responses)
  studio.send_comment(Comment(
    user_id="viewer_x", nickname="活跃观众", content="直播间好热闹啊！",
    event_type=EventType.DANMAKU,
  ))
  await asyncio.sleep(12)
  after = len(responses)
  print(f"   新增回复: {after - before} 条")
  has_prefix = any("[prefix_greet]" in r for r in responses[before:])
  print(f"   包含前缀问候: {'是' if has_prefix else '否'}")
  print(f"   剩余入场队列: {studio._event_responder.debug_state()['entry_pending']}")

  # ──────────────────────────────────────────────
  print("\n── 测试 11: TTL 过期丢弃 ──")
  studio.send_comment(Comment(
    user_id="expire_1", nickname="过期用户", content="",
    event_type=EventType.ENTRY,
  ))
  print(f"   入场前: {studio._event_responder.debug_state()['entry_pending']}")
  print("   等待 35 秒（超过 30s TTL）...")
  await asyncio.sleep(35)
  studio._event_responder._expire_stale()
  print(f"   入场后: {studio._event_responder.debug_state()['entry_pending']} (应该为 0)")

  # ──────────────────────────────────────────────
  print("\n" + "=" * 60)
  print("  测试完成")
  print("=" * 60)
  print(f"\n总回复数: {len(responses)}")
  print("\n所有回复:")
  for i, r in enumerate(responses):
    print(f"  {i+1}. {r[:100]}")

  await studio.stop()


if __name__ == "__main__":
  asyncio.run(main())
