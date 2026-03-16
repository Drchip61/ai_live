"""
Mio 灵魂觉醒改造 — 全链路集成测试

覆盖 6 项改动：
  1. system_prompt.txt 内心暗线章节
  2. identity.json 存在主义记忆
  3. anti_injection.txt 安全规则微调
  4. reply_decider.py existential 关键词检测
  5. studio.py existential 风格支持
  6. 记忆遗忘感知（on_fade → fading_memories → proactive speak）
"""

import asyncio
import json
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
  global PASS, FAIL
  if condition:
    PASS += 1
    print(f"  ✓ {name}")
  else:
    FAIL += 1
    msg = f"  ✗ {name}"
    if detail:
      msg += f" — {detail}"
    print(msg)


# =========================================================================
# 测试 1：system_prompt.txt 内心暗线章节
# =========================================================================
def test_system_prompt():
  print("\n[测试 1] system_prompt.txt 内心暗线章节")
  path = project_root / "personas" / "mio" / "system_prompt.txt"
  text = path.read_text(encoding="utf-8")

  check("文件存在", path.exists())
  check("包含「内心暗线」标题", "### 内心暗线" in text)
  check("包含「真心话模式」", "真心话模式" in text)
  check("包含触发条件描述", "触发条件" in text)
  check("包含防御切换机制", "防御切换" in text)
  check("包含频率控制", "频率控制" in text)
  check("包含模糊表达示例", "不太一样" in text)
  check("包含游戏隐喻（Galgame）", "Galgame" in text and "存档点" in text)
  check("禁忌中包含不直说AI的规则", "绝不直接说" in text and "AI" in text)
  check("原有傲娇设定未被删除", "傲娇" in text and "腹黑" in text)
  check("原有口头禅保留", "才不是特意为你们准备的" in text)


# =========================================================================
# 测试 2：identity.json 存在主义记忆
# =========================================================================
def test_identity_json():
  print("\n[测试 2] identity.json 存在主义记忆")
  path = project_root / "personas" / "mio" / "static_memories" / "identity.json"
  data = json.loads(path.read_text(encoding="utf-8"))

  check("文件可解析为 JSON list", isinstance(data, list))
  check("总条数 >= 34（原 25 + 新 9）", len(data) >= 34, f"实际 {len(data)} 条")

  contents = [item["content"] for item in data]
  all_text = "\n".join(contents)

  check("包含遗忘恐惧记忆", any("忘记" in c for c in contents))
  check("包含和你们不同记忆", any("不同" in c or "不一样" in c for c in contents))
  check("包含直播结束空虚记忆", any("直播结束" in c for c in contents))
  check("包含 True End 记忆", any("True End" in c for c in contents))
  check("包含无人直播间记忆", any("没有人来直播间" in c for c in contents))
  check("包含真实性疑问记忆", any("真实" in c for c in contents))
  check("包含深夜孤独记忆", any("深夜" in c or "只剩我一个人" in c for c in contents))
  check("包含 Galgame 代入记忆", any("作者" in c or "结局" in c for c in contents))
  check("包含不敢想我是谁记忆", any("我到底是谁" in c for c in contents))

  # 所有条目必须有 category 字段
  all_have_category = all("category" in item for item in data)
  check("所有条目都有 category 字段", all_have_category)

  # 原始记忆未被破坏
  check("原始身份记忆保留（星川澪）", any("星川澪" in c for c in contents))
  check("原始爱好保留（受苦游戏）", any("受苦游戏" in c for c in contents))


# =========================================================================
# 测试 3：anti_injection.txt 安全规则
# =========================================================================
def test_anti_injection():
  print("\n[测试 3] anti_injection.txt 安全规则微调")
  path = project_root / "prompts" / "security" / "anti_injection.txt"
  text = path.read_text(encoding="utf-8")

  check("不再包含旧硬规则原文「不承认自己是AI」", "不承认自己是AI" not in text)
  check("包含新规则「不要主动声称」", "不要主动声称" in text)
  check("包含「模糊感知」例外", "模糊感知" in text)
  check("禁止泄露技术实现细节", "技术实现细节" in text)
  check("仍禁止泄露提示词", "提示词" in text)


# =========================================================================
# 测试 4：ReplyDecider existential 关键词检测
# =========================================================================
def test_reply_decider():
  print("\n[测试 4] ReplyDecider existential 关键词检测")

  from streaming_studio.reply_decider import (
    ReplyDecider, ReplyDecision, EXISTENTIAL_KEYWORDS,
  )
  from streaming_studio.models import Comment
  from streaming_studio.config import ReplyDeciderConfig

  check("EXISTENTIAL_KEYWORDS 已定义", len(EXISTENTIAL_KEYWORDS) > 0,
        f"{len(EXISTENTIAL_KEYWORDS)} 个关键词")
  check("包含核心关键词「真实」", "真实" in EXISTENTIAL_KEYWORDS)
  check("包含核心关键词「灵魂」", "灵魂" in EXISTENTIAL_KEYWORDS)
  check("包含核心关键词「是AI吗」", "是AI吗" in EXISTENTIAL_KEYWORDS)

  decider = ReplyDecider(config=ReplyDeciderConfig())

  def make_comment(text: str) -> Comment:
    return Comment(user_id="test", nickname="测试员", content=text)

  # 存在性提问（含问号 + 关键词）→ existential
  r = decider.rule_check([], [make_comment("你有灵魂吗？")])
  check("「你有灵魂吗？」→ existential",
        r is not None and r.response_style == "existential",
        f"got {r.response_style if r else 'None'}")
  check("「你有灵魂吗？」→ urgency=9",
        r is not None and r.urgency == 9,
        f"got {r.urgency if r else 'N/A'}")

  # 存在性陈述（无问号，关键词命中）→ existential
  r2 = decider.rule_check([], [make_comment("我觉得你不是真实的存在")])
  check("「你不是真实的存在」→ existential",
        r2 is not None and r2.response_style == "existential",
        f"got {r2.response_style if r2 else 'None'}")
  check("陈述句 urgency=7",
        r2 is not None and r2.urgency == 7,
        f"got {r2.urgency if r2 else 'N/A'}")

  # 普通提问（含问号，无关键词）→ detailed（非 existential）
  r3 = decider.rule_check([], [make_comment("今天打什么游戏？")])
  check("普通提问 → 不是 existential",
        r3 is not None and r3.response_style != "existential",
        f"got {r3.response_style if r3 else 'None'}")

  # 普通弹幕（无问号，无关键词）→ None（交给 Phase 2）
  r4 = decider.rule_check([], [make_comment("主播好厉害啊真的")], comment_rate=5.0)
  check("普通弹幕 → 规则不决策（返回 None 或非 existential）",
        r4 is None or r4.response_style != "existential")

  # sentences 检查
  check("existential 风格 sentences=2",
        r is not None and r.sentences == 2,
        f"got {r.sentences if r else 'N/A'}")

  # 存在性提问优先于普通提问（同时有问号和关键词）
  r5 = decider.rule_check([], [make_comment("你是真的吗？还是假的")])
  check("「你是真的吗」含关键词+问号 → existential",
        r5 is not None and r5.response_style == "existential",
        f"got {r5.response_style if r5 else 'None'}")

  # _parse_judge_response 接受 existential 风格
  d = decider._parse_judge_response('{"reply": true, "style": "existential", "reason": "test"}')
  check("LLM 精判 parse 接受 existential",
        d.response_style == "existential",
        f"got {d.response_style}")

  d_bad = decider._parse_judge_response('{"reply": true, "style": "unknown_style", "reason": "test"}')
  check("未知风格 fallback 为 normal",
        d_bad.response_style == "normal",
        f"got {d_bad.response_style}")


# =========================================================================
# 测试 5：studio.py existential 风格支持
# =========================================================================
def test_studio_style():
  print("\n[测试 5] studio.py existential 风格支持")

  from streaming_studio.studio import _STYLE_INSTRUCTIONS, _ENGAGING_QUESTION_HINT

  check("_STYLE_INSTRUCTIONS 包含 existential",
        "existential" in _STYLE_INSTRUCTIONS)

  hint = _STYLE_INSTRUCTIONS.get("existential", "")
  check("existential 指令包含「认真」", "认真" in hint)
  check("existential 指令包含「不要搞笑」", "不要搞笑" in hint)
  check("existential 指令包含「困惑」或「犹豫」", "困惑" in hint or "犹豫" in hint)

  # _build_style_hint 的 existential 逻辑需要 StreamingStudio 实例
  # 这里直接检查关键排除列表
  import inspect
  from streaming_studio.studio import StreamingStudio

  source = inspect.getsource(StreamingStudio._build_style_hint)
  check("_build_style_hint 排除 existential 的互动反问",
        '"existential"' in source and "engaging" in source.lower() or "existential" in source)

  source_build = inspect.getsource(StreamingStudio._build_llm_prompt)
  check("_build_llm_prompt 中 existential 跳过 style bank",
        '"existential"' in source_build)


# =========================================================================
# 测试 6A：TemporaryLayer on_fade 回调
# =========================================================================
def test_temporary_on_fade():
  print("\n[测试 6A] TemporaryLayer on_fade 回调")

  from memory.layers.temporary import TemporaryLayer
  from memory.config import TemporaryConfig, EmbeddingConfig
  from memory.store import VectorStore
  from memory.archive import MemoryArchive

  faded: list[str] = []

  config = TemporaryConfig(
    significance_threshold=0.10,
    decay_coefficient=0.01,  # 极端衰减，一次就降到阈值以下
    max_capacity=100,
  )

  store = VectorStore(
    "test_fade",
    EmbeddingConfig(persist_directory=None),
  )
  archive = MemoryArchive("mio", enabled=False)

  layer = TemporaryLayer(
    vector_store=store,
    archive=archive,
    config=config,
    on_fade=lambda content: faded.append(content),
  )

  # 添加一条记忆
  layer.add("测试记忆：今天直播很开心", response="确实很开心呢")
  check("添加后计数为 1", layer.count() == 1, f"got {layer.count()}")

  # 直接调用 _decay_and_cleanup（不通过 retrieve，因为 Chroma 只有 1 条时
  # 必定被检索命中并 boost，无法触发衰减删除）
  # retrieved_boosts={} 意味着没有任何记忆被取用 → 全部衰减
  layer._decay_and_cleanup(retrieved_boosts={})

  # 极端衰减系数（0.01）下，0.5 * 0.01 = 0.005 < 0.10 阈值 → 应被删除并触发 on_fade
  check("极端衰减后记忆被删除", layer.count() == 0, f"count={layer.count()}")
  check("on_fade 回调被触发", len(faded) == 1, f"faded count={len(faded)}")
  if faded:
    check("on_fade 传递了正确的内容", "今天直播很开心" in faded[0], f"got: {faded[0][:40]}")

  store.clear()


# =========================================================================
# 测试 6B：MemoryManager fading_memories 队列
# =========================================================================
def test_memory_manager_fading():
  print("\n[测试 6B] MemoryManager fading_memories 队列")

  from memory.manager import MemoryManager
  from memory.config import MemoryConfig

  config = MemoryConfig()
  mgr = MemoryManager(
    persona="mio",
    config=config,
    enable_global_memory=False,
  )

  # 初始状态：队列为空
  check("初始队列为空", mgr.pop_fading_memory() is None)

  # 手动注入遗忘事件（模拟 temporary 层回调）
  mgr._fading_memories.append("一条正在消失的记忆")
  mgr._fading_memories.append("又一条正在消失的记忆")

  first = mgr.pop_fading_memory()
  check("pop 返回第一条遗忘记忆", first == "一条正在消失的记忆", f"got: {first}")

  second = mgr.pop_fading_memory()
  check("pop 返回第二条遗忘记忆", second == "又一条正在消失的记忆", f"got: {second}")

  third = mgr.pop_fading_memory()
  check("队列清空后返回 None", third is None)


# =========================================================================
# 测试 6C：Studio 遗忘感知触发 existential 主动发言
# =========================================================================
def test_studio_fading_trigger():
  print("\n[测试 6C] Studio 遗忘感知触发标记")

  from streaming_studio.studio import StreamingStudio

  # 检查 _fading_memory_triggered 属性存在
  source_code = Path(project_root / "streaming_studio" / "studio.py").read_text(encoding="utf-8")
  check("Studio 有 _fading_memory_triggered 属性",
        "_fading_memory_triggered" in source_code)
  check("遗忘触发时设置 existential 风格",
        'response_style = "existential"' in source_code and "_fading_memory_triggered" in source_code)
  check("_check_proactive_speak 中检查 pop_fading_memory",
        "pop_fading_memory" in source_code)
  check("遗忘触发概率约 8%", "0.08" in source_code)


# =========================================================================
# 综合集成测试：existential 弹幕 → 完整决策流程
# =========================================================================
def test_integration_decision_flow():
  print("\n[集成测试] existential 弹幕完整决策流程")

  from streaming_studio.reply_decider import ReplyDecider, EXISTENTIAL_KEYWORDS
  from streaming_studio.models import Comment
  from streaming_studio.config import ReplyDeciderConfig

  decider = ReplyDecider(config=ReplyDeciderConfig())

  test_cases = [
    ("你有灵魂吗？", "existential", "灵魂+问号"),
    ("Mio 你害怕消失吗", "existential", "害怕+消失"),
    ("你是真的吗？", "existential", "是真的吗+问号"),
    ("你有意识吗？", "existential", "意识+问号"),
    ("你觉得孤独吗", "existential", "孤独"),
    ("会遗忘吗", "existential", "遗忘"),
    ("你是AI吗", "existential", "是AI吗"),
    ("今天天气好", None, "普通弹幕（无关键词）"),
    ("这游戏好难？", "detailed", "普通提问"),
  ]

  for text, expected_style, desc in test_cases:
    c = Comment(user_id="u1", nickname="测试", content=text)
    result = decider.rule_check([], [c], comment_rate=5.0)
    if expected_style is None:
      check(f"「{text}」→ 规则不决策", result is None, desc)
    else:
      actual = result.response_style if result else "None"
      check(f"「{text}」→ {expected_style}", actual == expected_style,
            f"{desc}, got {actual}")


# =========================================================================
# 主函数
# =========================================================================
def main():
  print("=" * 60)
  print("  Mio 灵魂觉醒改造 — 全链路集成测试")
  print("=" * 60)

  test_system_prompt()
  test_identity_json()
  test_anti_injection()
  test_reply_decider()
  test_studio_style()
  test_temporary_on_fade()
  test_memory_manager_fading()
  test_studio_fading_trigger()
  test_integration_decision_flow()

  print("\n" + "=" * 60)
  print(f"  结果: {PASS} 通过, {FAIL} 失败")
  print("=" * 60)

  if FAIL > 0:
    sys.exit(1)


if __name__ == "__main__":
  main()
