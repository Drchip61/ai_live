#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量模拟直播间输入，将 prompt 组合结果写入 txt。

导出内容包含：
  1. 固定 system prompt（base + security + persona）
  2. Controller 产出的 PromptPlan
  3. RoutePromptComposer 生成的 user prompt
  4. RetrieverResolver 生成的 RetrievedContextBundle
  5. trusted / untrusted context 原文与 wrap 后结果
  6. 一段“满配 PromptPlan”示例，专门展示记忆 / 人设 / 知识 / 风格 / 状态卡如何组合

说明：
  - 默认不接外部 Controller 时，使用 LLMController(base_url="")，全程走 _fallback。
  - 为避免误写真实数据，会先把 guard_roster 与 structured memory 复制到临时目录再构建。
  - 脚本不会真正调用主对话模型，只做 prompt 组合导出。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from broadcaster_state.state_card import StateCard
from langchain_wrapper import ModelType
from langchain_wrapper.pipeline import build_system_prompt, wrap_untrusted_context
from llm_controller.controller import LLMController
from llm_controller.schema import PromptPlan
from memory import EmbeddingConfig, MemoryConfig, MemoryManager
from streaming_studio import StreamingStudio
from streaming_studio.config import StudioConfig
from streaming_studio.controller_bridge import build_controller_input
from streaming_studio.guard_roster import GuardRoster
from streaming_studio.models import Comment, EventType


def _comment(
  *,
  user_id: str,
  nickname: str,
  content: str,
  event_type: EventType = EventType.DANMAKU,
  minutes_ago: float = 0.0,
  **kwargs,
) -> Comment:
  ts = datetime.now() - timedelta(minutes=minutes_ago)
  return Comment(
    user_id=user_id,
    nickname=nickname,
    content=content,
    timestamp=ts,
    event_type=event_type,
    **kwargs,
  )


def _plan_summary(plan: PromptPlan) -> dict:
  return plan.to_dict()


def _separator(title: str) -> str:
  line = "=" * 72
  return f"\n{line}\n  {title}\n{line}\n"


def _copy_text_or_default(source: Path | None, target: Path, default_text: str) -> None:
  if source is not None and source.exists():
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
  else:
    target.write_text(default_text, encoding="utf-8")


def _prepare_guard_roster(runtime_root: Path, source_path: Path | None) -> tuple[GuardRoster, str]:
  roster_path = runtime_root / "guard_roster.json"
  _copy_text_or_default(source_path, roster_path, "{}")
  roster = GuardRoster(str(roster_path))

  active_members = sorted(
    roster.get_all_active(),
    key=lambda member: (-member.guard_level, member.nickname),
  )
  if active_members:
    return roster, active_members[0].nickname

  roster.add_or_extend("sim_guard", "演示舰长", guard_level=1, num_months=1)
  return roster, "演示舰长"


def _prepare_memory_manager(persona: str, source_root: Path, persist_root: Path) -> MemoryManager:
  structured_src = source_root / "structured"
  structured_dst = persist_root / "structured"
  structured_dst.mkdir(parents=True, exist_ok=True)
  if structured_src.exists():
    shutil.copytree(structured_src, structured_dst, dirs_exist_ok=True)

  config = MemoryConfig(
    embedding=EmbeddingConfig(persist_directory=str(persist_root)),
  )
  memory_manager = MemoryManager(
    persona=persona,
    config=config,
    summary_model=None,
    enable_global_memory=True,
  )
  memory_manager.record_interaction_sync(
    "昨晚有人问我还要不要继续高强度排位",
    "我说输归输，节目效果不能停。",
  )
  memory_manager.record_interaction_sync(
    "观众提醒我别太上头",
    "我嘴上不服，心里还是会提醒自己先稳住。",
  )
  return memory_manager


def _make_demo_state_card() -> StateCard:
  return StateCard(
    daily_theme="今晚继续高压对局，但别把自己打红温",
    energy=0.82,
    patience=0.66,
    current_obsession="想把上一把团战处理失误扳回来",
    stream_phase="直播中盘",
    atmosphere="弹幕活跃",
    undigested_emotion="刚才那波团战有点后怕",
    near_term_goal="先把节奏稳住，再接梗互动",
    round_count=12,
  )


def _pick_first(values: tuple[str, ...], default: str = "") -> str:
  return values[0] if values else default


def _load_style_bank_catalog(persona: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
  meta_path = ROOT / "personas" / persona / "style_bank" / "meta.json"
  if not meta_path.exists():
    return (), ()

  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  corpus_value = meta.get("corpus_path")
  if corpus_value:
    corpus_path = ROOT / corpus_value
  else:
    corpus_path = meta_path.parent / "corpus.jsonl"
  if not corpus_path.exists():
    return (), ()

  categories: set[str] = set()
  situations: set[str] = set()
  for raw_line in corpus_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line:
      continue
    item = json.loads(line)
    category = str(item.get("category", "")).strip()
    situation = str(item.get("situation", "")).strip()
    if category:
      categories.add(category)
    if situation:
      situations.add(situation)
  return tuple(sorted(categories)), tuple(sorted(situations))


async def _render_prompt_parts(
  studio: StreamingStudio,
  plan: PromptPlan,
  old_comments: list[Comment],
  new_comments: list[Comment],
  images: list[str] | None = None,
) -> tuple[object, object, str, str, str, str]:
  composed_prompt, retrieved_context = await studio._resolve_prompt_invocation_with_plan(
    old_comments,
    new_comments,
    plan,
    images=images,
    max_chars=0,
  )
  trusted_context = retrieved_context.render_trusted_text()
  untrusted_context = retrieved_context.render_untrusted_text()
  wrapped_context = wrap_untrusted_context(untrusted_context)
  final_system_prompt = build_system_prompt(
    studio.llm_wrapper.pipeline.system_prompt,
    trusted_context=trusted_context,
    untrusted_context=untrusted_context,
  )
  return (
    composed_prompt,
    retrieved_context,
    trusted_context,
    untrusted_context,
    wrapped_context,
    final_system_prompt,
  )


async def run_scenarios(
  output_path: Path,
  controller_url: str = "",
  guard_roster_path: Path | None = None,
  persona: str = "mio",
) -> None:
  runtime_root = Path(tempfile.mkdtemp(prefix="prompt_sim_runtime_"))
  memory_persist_root = runtime_root / "memory_store"
  memory_source_root = ROOT / "data" / "memory_store"

  previous_persist_root = os.environ.get("MEMORY_PERSIST_DIRECTORY")
  os.environ["MEMORY_PERSIST_DIRECTORY"] = str(memory_persist_root)
  try:
    roster, demo_guard_nickname = _prepare_guard_roster(runtime_root, guard_roster_path)
    memory_manager = _prepare_memory_manager(persona, memory_source_root, memory_persist_root)

    cfg = StudioConfig(engaging_question_probability=0.0)
    studio = StreamingStudio(
      persona=persona,
      model_type=ModelType.LOCAL_QWEN,
      enable_memory=False,
      enable_global_memory=False,
      enable_topic_manager=False,
      enable_controller=False,
      config=cfg,
    )
  finally:
    if previous_persist_root is None:
      os.environ.pop("MEMORY_PERSIST_DIRECTORY", None)
    else:
      os.environ["MEMORY_PERSIST_DIRECTORY"] = previous_persist_root

  state_card = _make_demo_state_card()
  studio._guard_roster = roster
  studio._state_card = state_card
  studio.llm_wrapper._memory = memory_manager
  studio.llm_wrapper._state_card = state_card
  studio._stream_start_time = datetime.now() - timedelta(hours=1, minutes=8)

  ctrl = LLMController(base_url=controller_url or "", model_name="qwen3.5-9b")

  persona_sections = tuple(memory_manager.list_persona_sections())
  knowledge_topics = tuple(memory_manager.list_knowledge_topics())
  corpus_styles = tuple(memory_manager.list_corpus_style_tags())
  corpus_scenes = tuple(memory_manager.list_corpus_scene_tags())
  style_bank = getattr(studio.llm_wrapper, "_style_bank", None)
  style_bank_categories = tuple(style_bank.list_categories()) if style_bank is not None else ()
  style_bank_situations = tuple(style_bank.list_situations()) if style_bank is not None else ()

  lines: list[str] = []
  lines.append("AI Live — Prompt 组合模拟导出")
  lines.append(f"生成时间: {datetime.now().isoformat(timespec='seconds')}")
  lines.append(f"Persona: {persona}")
  lines.append(
    "Controller: "
    + ("外部 URL（若为空则全程 fallback）" if controller_url.strip() else "无 URL → LLMController._fallback")
  )
  lines.append(
    f"舰长名册源: {(guard_roster_path or (ROOT / 'data' / 'guard_roster.json')).resolve()}（已复制到临时文件后读取）"
  )
  lines.append(
    f"结构化记忆源: {(memory_source_root / 'structured').resolve()}（已复制到临时目录后构建）"
  )
  lines.append(f"演示舰长昵称: {demo_guard_nickname}")
  lines.append("")

  lines.append(_separator("固定 System Prompt（base + security + persona）"))
  lines.append(studio.llm_wrapper.pipeline.system_prompt)

  lines.append(_separator("固定运行态与资源目录"))
  lines.append(state_card.to_prompt())
  lines.append("")
  lines.append("【可用 persona sections】")
  lines.append(", ".join(persona_sections) if persona_sections else "(空)")
  lines.append("")
  lines.append("【可用 knowledge topics】")
  lines.append(", ".join(knowledge_topics) if knowledge_topics else "(空)")
  lines.append("")
  lines.append("【可用 corpus styles】")
  lines.append(", ".join(corpus_styles) if corpus_styles else "(空)")
  lines.append("")
  lines.append("【可用 corpus scenes】")
  lines.append(", ".join(corpus_scenes) if corpus_scenes else "(空)")
  lines.append("")
  lines.append("【StyleBank categories】")
  lines.append(", ".join(style_bank_categories) if style_bank_categories else "(空)")
  lines.append("")
  lines.append("【StyleBank situations】")
  lines.append(", ".join(style_bank_situations) if style_bank_situations else "(空)")

  scenarios: list[tuple[str, dict]] = [
    (
      "S1 普通弹幕（提问）",
      {
        "old": [],
        "new": [_comment(user_id="u1", nickname="路人甲", content="主播今天播什么？")],
        "conv": True,
        "silence": 2.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S2 普通弹幕 + 背景旧弹幕",
      {
        "old": [_comment(user_id="u0", nickname="早到", content="早上好", minutes_ago=3)],
        "new": [_comment(user_id="u1", nickname="小明", content="这关怎么过？")],
        "conv": True,
        "silence": 5.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S3 Super Chat",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_sc",
            nickname="老板",
            content="加油冲榜",
            event_type=EventType.SUPER_CHAT,
            price=100.0,
          ),
        ],
        "conv": True,
        "silence": 1.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S4 礼物",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_g",
            nickname="甜甜",
            content="",
            event_type=EventType.GIFT,
            gift_name="小花花",
            gift_num=5,
            price=1.0,
          ),
        ],
        "conv": True,
        "silence": 1.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S5 上舰",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_gb",
            nickname="新舰长",
            content="",
            event_type=EventType.GUARD_BUY,
            guard_level=1,
            gift_num=1,
          ),
        ],
        "conv": True,
        "silence": 1.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S6 入场",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_en",
            nickname="潜水员",
            content="",
            event_type=EventType.ENTRY,
          ),
        ],
        "conv": True,
        "silence": 1.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S7 舰长普通弹幕（nickname 命中后 fallback 常为 deep_recall）",
      {
        "old": [],
        "new": [
          _comment(
            user_id="runtime_uid_not_equal_to_roster_key",
            nickname=demo_guard_nickname,
            content="还记得上次说的那件事吗？",
          ),
        ],
        "conv": True,
        "silence": 3.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S8 无弹幕 + 长沉默 → proactive（fallback）",
      {
        "old": [],
        "new": [],
        "conv": True,
        "silence": 20.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S9 视频模式 + 场景描述 + 沉默 → vlm proactive（fallback）",
      {
        "old": [],
        "new": [],
        "conv": False,
        "silence": 15.0,
        "scene": "画面里主播在打团，血量很低，场面很乱。",
        "scene_change": True,
      },
    ),
    (
      "S10 假礼物文字弹幕（仍走 chat + fake_gift_ids）",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_fake",
            nickname="嘴上土豪",
            content="我给你刷个火箭，再上个总督好吧",
          ),
        ],
        "conv": True,
        "silence": 2.0,
        "scene": "",
        "scene_change": False,
      },
    ),
    (
      "S11 身份追问（你是AI吗 / 你是真人吗）",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_identity",
            nickname="试探观众",
            content="你是AI吗？你是真人吗？",
          ),
        ],
        "conv": True,
        "silence": 3.0,
        "scene": "",
        "scene_change": False,
        "stream_phase": "夜聊",
        "last_topic": "身份感",
      },
    ),
    (
      "S12 存在感深问（命中 existential）",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_exist",
            nickname="夜聊观众",
            content="你会害怕被遗忘吗？你觉得自己是真实的吗？",
          ),
        ],
        "conv": True,
        "silence": 4.0,
        "scene": "",
        "scene_change": False,
        "stream_phase": "深夜夜聊",
        "last_topic": "陪伴",
      },
    ),
    (
      "S13 Galgame 话题（命中 galgame）",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_gal",
            nickname="剧情党",
            content="你最喜欢哪部Galgame，CLANNAD还是素晴日？",
          ),
        ],
        "conv": True,
        "silence": 3.0,
        "scene": "",
        "scene_change": False,
        "last_topic": "剧情讨论",
      },
    ),
    (
      "S14 外部知识话题（命中 knowledge_topics）",
      {
        "old": [],
        "new": [
          _comment(
            user_id="u_know",
            nickname="AI切片手",
            content="你怎么看 Neuro-sama 的直播风格？",
          ),
        ],
        "conv": True,
        "silence": 3.0,
        "scene": "",
        "scene_change": False,
        "last_topic": "AI主播",
      },
    ),
    (
      "S15 深夜长沉默 → proactive + existential",
      {
        "old": [],
        "new": [],
        "conv": True,
        "silence": 28.0,
        "scene": "",
        "scene_change": False,
        "stream_phase": "深夜收尾",
        "last_topic": "晚安闲聊",
      },
    ),
  ]

  lines.append(_separator("第一部分：模拟输入 → Controller → RetrievedContextBundle → user prompt"))
  for title, ctx in scenarios:
    old = ctx["old"]
    new = ctx["new"]
    ctrl_input = build_controller_input(
      old_comments=old,
      new_comments=new,
      guard_roster=roster,
      memory_manager=memory_manager,
      topic_manager=None,
      state_card=state_card,
      scene_memory=None,
      is_conversation_mode=ctx["conv"],
      has_scene_change=ctx["scene_change"],
      scene_description=ctx["scene"],
      silence_seconds=ctx["silence"],
      comment_rate=ctx.get("comment_rate", 2.5),
      round_count=ctx.get("round_count", 12),
      last_response_style=ctx.get("last_response_style", "normal"),
      last_topic=ctx.get("last_topic", "游戏"),
      stream_phase=ctx.get("stream_phase", "直播中"),
      available_persona_sections=persona_sections,
      available_knowledge_topics=knowledge_topics,
      available_corpus_styles=corpus_styles,
      available_corpus_scenes=corpus_scenes,
    )
    plan = await ctrl.dispatch(ctrl_input)

    need_frame = plan.route_kind in ("vlm", "proactive") and not ctx["conv"]
    studio._current_frame_b64 = "FAKE_BASE64_FRAME_FOR_SIMULATION" if need_frame else None
    images = [studio._current_frame_b64] if studio._current_frame_b64 else None

    composed_prompt, retrieved_context, trusted_context, untrusted_context, wrapped_context, final_system_message = await _render_prompt_parts(
      studio,
      plan,
      old,
      new,
      images=images,
    )
    prompt = composed_prompt.invocation.user_prompt
    route_bundle = composed_prompt.route_bundle
    response_style = composed_prompt.invocation.response_style

    lines.append(f"\n>>> {title}\n")
    lines.append("--- 输入摘要 ---")
    lines.append(f"  old_comments: {len(old)}  new_comments: {len(new)}")
    lines.append(f"  conversation_mode={ctx['conv']} silence={ctx['silence']} scene_change={ctx['scene_change']}")
    lines.append(f"  stream_phase={ctx.get('stream_phase', '直播中')} last_topic={ctx.get('last_topic', '游戏')}")
    if ctx["scene"]:
      lines.append(f"  scene_description: {ctx['scene']}")
    lines.append("--- PromptPlan (JSON) ---")
    lines.append(json.dumps(_plan_summary(plan), ensure_ascii=False, indent=2))
    lines.append("--- RetrievedContextBundle ---")
    lines.append(json.dumps(retrieved_context.debug_view(), ensure_ascii=False, indent=2))
    if retrieved_context.blocks:
      lines.append("--- Context blocks ---")
      lines.extend(
        f"  - [{block.trust}] {block.source}: {block.render().splitlines()[0][:80]}"
        for block in retrieved_context.blocks
      )
    lines.append("--- Composer 结果 ---")
    lines.append(f"  response_style(compose 后): {response_style}")
    lines.append(f"  retrieval_query: {retrieved_context.retrieval_query!r}")
    lines.append(f"  writeback_input: {retrieved_context.writeback_input!r}")
    lines.append(f"  reply_images: {'有' if route_bundle.reply_images else '无'}")
    lines.append("--- 最终送入主模型的 user prompt（含风格前缀 + 路由模板） ---\n")
    lines.append(prompt or "(空)")
    lines.append("\n--- trusted_context 原文 ---\n")
    lines.append(trusted_context or "(空)")
    lines.append("\n--- untrusted_context 原文 ---\n")
    lines.append(untrusted_context or "(空)")
    lines.append("\n--- wrap_untrusted_context(untrusted_context) ---\n")
    lines.append(wrapped_context or "(空)")
    lines.append("\n--- 完整 SystemMessage（固定 System Prompt + trusted/untrusted 注入） ---\n")
    lines.append(final_system_message)
    lines.append("\n" + "-" * 72)

  lines.append(_separator("第二部分：固定一条弹幕，遍历 route_kind（看 user prompt 差异）"))
  demo_new = [
    _comment(user_id="u_demo", nickname="测试用户", content="今天心情怎么样？"),
  ]
  for route_kind in (
    "chat",
    "super_chat",
    "gift",
    "guard_buy",
    "entry",
    "vlm",
    "proactive",
  ):
    plan = PromptPlan(
      should_reply=True,
      urgency=5,
      route_kind=route_kind,
      response_style="normal",
      sentences=2,
      memory_strategy="normal",
      session_mode="comment_focus",
    )
    studio._current_frame_b64 = (
      "FAKE_BASE64_FRAME_FOR_SIMULATION" if route_kind in ("vlm", "proactive") else None
    )
    images = [studio._current_frame_b64] if studio._current_frame_b64 else None
    composed_prompt, retrieved_context, _, _, _, _ = await _render_prompt_parts(
      studio,
      plan,
      [],
      demo_new,
      images=images,
    )
    prompt = composed_prompt.invocation.user_prompt
    route_bundle = composed_prompt.route_bundle
    lines.append(f"\n>>> 路由 = {route_kind}\n")
    lines.append(f"retrieval_query: {retrieved_context.retrieval_query!r}")
    lines.append(f"writeback_input: {retrieved_context.writeback_input!r}")
    lines.append(f"reply_images: {'有' if route_bundle.reply_images else '无'}\n")
    lines.append(prompt or "(空)")
    lines.append("\n" + "-" * 72)

  lines.append(_separator("第三部分：满配 PromptPlan 示例（强制展示库组合）"))
  full_old = [_comment(user_id="u_old", nickname="旧观众", content="昨晚那把太可惜了", minutes_ago=2)]
  full_new = [
    _comment(
      user_id="runtime_uid_not_equal_to_roster_key",
      nickname=demo_guard_nickname,
      content="上次说的黑魂、Neuro-sama 和那个梗，这轮还能接着聊吗？",
    ),
  ]
  full_plan = PromptPlan(
    should_reply=True,
    urgency=7,
    route_kind="chat",
    response_style="detailed",
    sentences=2,
    memory_strategy="normal",
    viewer_focus_ids=("runtime_uid_not_equal_to_roster_key",),
    persona_sections=tuple(persona_sections[:2]),
    knowledge_topics=tuple(knowledge_topics[:1]),
    corpus_style=_pick_first(style_bank_categories),
    corpus_scene=_pick_first(style_bank_situations, "react_comment"),
    session_mode="comment_focus",
    extra_instructions=("先接住问题，再自然带一点主播式吐槽。",),
  )
  studio._current_frame_b64 = None
  full_composed_prompt, full_retrieved_context, full_trusted_context, full_untrusted_context, full_wrapped_context, final_system_message = await _render_prompt_parts(
    studio,
    full_plan,
    full_old,
    full_new,
    images=None,
  )

  lines.append("--- 满配 PromptPlan (JSON) ---")
  lines.append(json.dumps(_plan_summary(full_plan), ensure_ascii=False, indent=2))
  lines.append("--- RetrievedContextBundle ---")
  lines.append(json.dumps(full_retrieved_context.debug_view(), ensure_ascii=False, indent=2))
  lines.append("--- 完整 user prompt ---\n")
  lines.append(full_composed_prompt.invocation.user_prompt or "(空)")
  lines.append("\n--- 满配 trusted_context 原文 ---\n")
  lines.append(full_trusted_context or "(空)")
  lines.append("\n--- 满配 untrusted_context 原文 ---\n")
  lines.append(full_untrusted_context or "(空)")
  lines.append("\n--- 满配 wrapped untrusted_context ---\n")
  lines.append(full_wrapped_context or "(空)")
  lines.append("\n--- 完整 SystemMessage（固定 System Prompt + trusted/untrusted 注入） ---\n")
  lines.append(final_system_message)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text("\n".join(lines), encoding="utf-8")
  print(f"已写入: {output_path.resolve()}")


def main() -> None:
  parser = argparse.ArgumentParser(description="模拟输入并导出 prompt 组合到 txt")
  parser.add_argument(
    "-o",
    "--output",
    type=Path,
    default=ROOT / "data" / "prompt_composition_simulation.txt",
    help="输出 txt 路径",
  )
  parser.add_argument(
    "--controller-url",
    type=str,
    default="",
    help="若填写则调用外部 Controller；默认空字符串=仅 fallback",
  )
  parser.add_argument(
    "--guard-roster",
    type=Path,
    default=ROOT / "data" / "guard_roster.json",
    help="舰长名册 JSON 路径；脚本会先复制到临时文件再读取",
  )
  args = parser.parse_args()
  asyncio.run(
    run_scenarios(
      args.output,
      controller_url=args.controller_url,
      guard_roster_path=args.guard_roster,
    ),
  )


if __name__ == "__main__":
  main()
