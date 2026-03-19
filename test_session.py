"""
FocusSession 聚焦会话系统 — 全面单元测试

覆盖范围：
  1. 高质量弹幕判定（_is_quality_comment）
  2. CommentSession 生命周期（开启 → 续延 → 达轮数结束 / 无关结束）
  3. VideoSession 生命周期（开启 → 弹幕打断 → 高质量切换）
  4. Prompt 生成（CommentSession / VideoSession / 被打断状态）
  5. 相关性检测（_has_relevant_comment）
  6. 超时结束
  7. Session 互斥（已有 session 时不重复开启）
  8. debug_state 正确性
  9. 配置禁用
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from streaming_studio.models import Comment, EventType
from streaming_studio.config import SessionConfig
from streaming_studio.session import SessionManager, SessionType, FocusSession


def _c(content: str, **kwargs) -> Comment:
  """快速创建普通弹幕"""
  return Comment(
    user_id=kwargs.get("user_id", "u1"),
    nickname=kwargs.get("nickname", "测试用户"),
    content=content,
    priority=kwargs.get("priority", False),
    event_type=kwargs.get("event_type", EventType.DANMAKU),
    price=kwargs.get("price", 0.0),
  )


def _sc(content: str, price: float = 50.0) -> Comment:
  """快速创建 SC"""
  return Comment(
    user_id="sc_user", nickname="SC用户", content=content,
    event_type=EventType.SUPER_CHAT, price=price,
  )


# ──────────────────────────────────────────────
# 1. 高质量弹幕判定
# ──────────────────────────────────────────────

def test_quality_sc_always_high():
  """SC 永远是高质量"""
  sm = SessionManager()
  assert sm._is_quality_comment(_sc("加油", 10)), "SC 应该是高质量"
  assert sm._is_quality_comment(_sc("", 100)), "SC 内容为空也是高质量"
  print("  [PASS] SC 永远高质量")


def test_quality_priority_always_high():
  """priority 弹幕永远高质量"""
  sm = SessionManager()
  c = _c("a", priority=True)
  assert sm._is_quality_comment(c), "priority 弹幕应该是高质量"
  print("  [PASS] priority 弹幕永远高质量")


def test_quality_noise_excluded():
  """噪声弹幕排除"""
  sm = SessionManager()
  noise_samples = [
    "哈哈哈", "666", "草", "yyds", "nb", "233", "awsl",
    "来了", "你好", "hello", "主播好", "👍👏🔥",
  ]
  for text in noise_samples:
    assert not sm._is_quality_comment(_c(text)), f"「{text}」不应是高质量"
  print("  [PASS] 噪声弹幕全部排除")


def test_quality_repetitive_excluded():
  """纯重复字符排除"""
  sm = SessionManager()
  assert not sm._is_quality_comment(_c("啊啊啊啊啊")), "纯重复不应高质量"
  assert not sm._is_quality_comment(_c("666666666")), "纯重复不应高质量"
  print("  [PASS] 纯重复字符排除")


def test_quality_question():
  """提问类判定"""
  sm = SessionManager()
  assert sm._is_quality_comment(_c("你觉得这个游戏好玩吗？")), "带问号长问句应高质量"
  assert sm._is_quality_comment(_c("为什么不能联机？")), "有实质内容的问句应高质量"
  assert not sm._is_quality_comment(_c("？？")), "纯问号不应高质量（被噪声过滤）"
  assert not sm._is_quality_comment(_c("啥？")), "太短的问句不应高质量"
  print("  [PASS] 提问类判定正确")


def test_quality_opinion():
  """观点/讨论类判定"""
  sm = SessionManager()
  assert sm._is_quality_comment(_c("我觉得这个结局挺好的")), "含观点标记词且够长应高质量"
  assert sm._is_quality_comment(_c("推荐大家去玩一下这个游戏")), "推荐类应高质量"
  assert sm._is_quality_comment(_c("为什么这个角色这么强啊")), "为什么类应高质量"
  assert not sm._is_quality_comment(_c("觉得")), "太短不应高质量"
  print("  [PASS] 观点/讨论类判定正确")


def test_quality_long_content():
  """纯长度触发"""
  sm = SessionManager()
  assert sm._is_quality_comment(_c("这个场景让我想起了之前玩的一个游戏")), ">=12字应高质量"
  assert not sm._is_quality_comment(_c("好好看")), "太短不应高质量"
  print("  [PASS] 长内容触发判定正确")


def test_quality_non_danmaku_excluded():
  """非弹幕事件不触发（GIFT/ENTRY 等）"""
  sm = SessionManager()
  gift = _c("", event_type=EventType.GIFT)
  entry = _c("", event_type=EventType.ENTRY)
  guard = _c("", event_type=EventType.GUARD_BUY)
  assert not sm._is_quality_comment(gift), "GIFT 不应触发"
  assert not sm._is_quality_comment(entry), "ENTRY 不应触发"
  assert not sm._is_quality_comment(guard), "GUARD_BUY 不应触发"
  print("  [PASS] 非弹幕事件排除")


def test_find_quality_comment():
  """find_quality_comment 从列表中找到第一条高质量弹幕"""
  sm = SessionManager()
  comments = [
    _c("哈哈"),
    _c("666"),
    _c("这个boss战的机制设计真的很有意思"),
    _c("确实"),
  ]
  result = sm.find_quality_comment(comments)
  assert result is not None, "应能找到高质量弹幕"
  assert "boss" in result.content, f"应找到boss战那条，实际找到: {result.content}"
  print("  [PASS] find_quality_comment 正确")


def test_find_quality_comment_empty():
  """无高质量弹幕时返回 None"""
  sm = SessionManager()
  comments = [_c("哈哈"), _c("666"), _c("nb")]
  assert sm.find_quality_comment(comments) is None, "全是噪声应返回 None"
  assert sm.find_quality_comment([]) is None, "空列表应返回 None"
  print("  [PASS] 无高质量弹幕返回 None")


# ──────────────────────────────────────────────
# 2. CommentSession 生命周期
# ──────────────────────────────────────────────

def test_comment_session_open():
  """高质量弹幕触发 CommentSession"""
  sm = SessionManager()
  quality = _c("你觉得这个游戏的boss设计怎么样？")
  result = sm.evaluate_new_session([quality], [])
  assert result is True, "应开启 session"
  assert sm.is_active, "session 应为活跃"
  assert sm.session_type == SessionType.COMMENT, "应为 CommentSession"
  assert sm.session.anchor_text == quality.content.strip()
  assert sm.session.round_count == 0
  print("  [PASS] CommentSession 正确开启")


def test_comment_session_round_increment():
  """回复后轮次递增"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个结局设计怎么样")], [])

  sm.update_after_response(
    "#[wave][happy][joy] 我觉得结局很棒 / 結末は素晴らしいと思う",
    [_c("确实，结局设计很好")],
  )
  assert sm.session.round_count == 1
  assert len(sm.session.history) == 1
  assert "我觉得结局很棒" in sm.session.history[0], f"历史记录应去掉标签: {sm.session.history}"
  print("  [PASS] 轮次递增 + 历史记录正确")


def test_comment_session_end_max_rounds():
  """达到最大轮数自动结束"""
  config = SessionConfig(comment_session_max_rounds=2)
  sm = SessionManager(config=config)
  anchor = "你觉得这个游戏的boss设计怎么样"
  sm.evaluate_new_session([_c(anchor)], [])

  sm.update_after_response("回复1", [_c("确实，boss设计很好")])
  assert sm.is_active, "第1轮后应还在"

  sm.update_after_response("回复2", [_c("那个招式太难了")])
  assert not sm.is_active, "第2轮后应自动结束"
  print("  [PASS] 达到最大轮数自动结束")


def test_comment_session_end_stale():
  """连续无相关弹幕自动结束"""
  config = SessionConfig(stale_rounds_to_end=2, comment_session_max_rounds=10)
  sm = SessionManager(config=config)
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])

  # 第1轮：无相关弹幕 → stale_rounds=1
  sm.update_after_response("回复1", [_c("我去吃饭了")])
  assert sm.is_active, "第1轮无关后应还在"
  assert sm.session.stale_rounds == 1

  # 第2轮：继续无相关 → stale_rounds=2 → 结束
  sm.update_after_response("回复2", [_c("明天见")])
  assert not sm.is_active, "连续2轮无关应结束"
  print("  [PASS] 连续无相关弹幕自动结束")


def test_comment_session_stale_reset():
  """有相关弹幕时重置 stale 计数"""
  config = SessionConfig(stale_rounds_to_end=2, comment_session_max_rounds=10)
  sm = SessionManager(config=config)
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])

  # 第1轮：无相关
  sm.update_after_response("回复1", [_c("我去吃饭了")])
  assert sm.session.stale_rounds == 1

  # 第2轮：有相关（包含"boss"和"设计"）→ stale 重置
  sm.update_after_response("回复2", [_c("boss设计确实厉害")])
  assert sm.is_active, "有相关弹幕应续延"
  assert sm.session.stale_rounds == 0, "stale 应重置为 0"
  print("  [PASS] 相关弹幕重置 stale 计数")


def test_comment_session_no_duplicate_open():
  """已有 CommentSession 时不重复开启新的"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])
  original_anchor = sm.session.anchor_text

  result = sm.evaluate_new_session([_c("你们推荐什么好看的动漫吗")], [])
  assert result is False, "已有 CommentSession 不应开新的"
  assert sm.session.anchor_text == original_anchor, "anchor 不应变"
  print("  [PASS] 已有 CommentSession 不重复开启")


# ──────────────────────────────────────────────
# 3. VideoSession 生命周期
# ──────────────────────────────────────────────

def test_video_session_open():
  """场景变化 + 无弹幕 + 沉默 → 开启 VideoSession"""
  sm = SessionManager()
  result = sm.evaluate_new_session(
    [], [],
    has_scene_change=True, silence_seconds=20.0,
  )
  assert result is True, "应开启 VideoSession"
  assert sm.session_type == SessionType.VIDEO
  assert sm.session.max_rounds == 5
  print("  [PASS] VideoSession 正确开启")


def test_video_session_not_open_with_comments():
  """有弹幕时不开启 VideoSession"""
  sm = SessionManager()
  result = sm.evaluate_new_session(
    [_c("随便说说")], [],
    has_scene_change=True, silence_seconds=20.0,
  )
  # 弹幕不够高质量也不开 CommentSession，什么都不开
  assert not sm.is_active, "有弹幕不应开 VideoSession"
  print("  [PASS] 有弹幕不开 VideoSession")


def test_video_session_not_open_insufficient_silence():
  """沉默时间不够不开启 VideoSession"""
  sm = SessionManager()
  result = sm.evaluate_new_session(
    [], [], has_scene_change=True, silence_seconds=5.0,
  )
  assert result is False, "沉默不够不应开启"
  assert not sm.is_active
  print("  [PASS] 沉默不够不开 VideoSession")


def test_video_session_interrupt_by_normal_comment():
  """普通弹幕打断 VideoSession（标记 interrupted，不结束）"""
  sm = SessionManager()
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)

  sm.on_comment_arrived([_c("主播在看什么")])
  assert sm.is_active, "普通弹幕不应结束 VideoSession"
  assert sm.session.interrupted is True, "应标记为 interrupted"
  print("  [PASS] 普通弹幕打断标记 interrupted")


def test_video_session_interrupt_cleared_when_no_comments():
  """无弹幕时清除 interrupted 标记"""
  sm = SessionManager()
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)
  sm.on_comment_arrived([_c("主播在看什么")])
  assert sm.session.interrupted is True

  sm.on_comment_arrived([])
  assert sm.session.interrupted is False, "无弹幕应清除 interrupted"
  print("  [PASS] 无弹幕时清除 interrupted")


def test_video_session_switch_to_comment_by_quality():
  """高质量弹幕打断 VideoSession → 切换为 CommentSession"""
  sm = SessionManager()
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)
  assert sm.session_type == SessionType.VIDEO

  quality = _c("你觉得这段剧情的设计怎么样？")
  result = sm.evaluate_new_session([quality], [])
  assert result is True, "应切换 session"
  assert sm.session_type == SessionType.COMMENT, "应变成 CommentSession"
  assert sm.session.anchor_text == quality.content.strip()
  print("  [PASS] 高质量弹幕切换 VideoSession → CommentSession")


def test_video_session_end_max_rounds():
  """VideoSession 达到最大轮数结束"""
  config = SessionConfig(video_session_max_rounds=2)
  sm = SessionManager(config=config)
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)

  sm.update_after_response("画面1", [])
  assert sm.is_active

  sm.update_after_response("画面2", [])
  assert not sm.is_active, "达到最大轮数应结束"
  print("  [PASS] VideoSession 达到最大轮数结束")


def test_on_comment_arrived_noop_for_comment_session():
  """on_comment_arrived 对 CommentSession 无效"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])
  sm.on_comment_arrived([_c("哈哈")])
  assert sm.session.interrupted is False, "CommentSession 不应被标记 interrupted"
  print("  [PASS] on_comment_arrived 对 CommentSession 无效")


# ──────────────────────────────────────────────
# 4. Prompt 生成
# ──────────────────────────────────────────────

def test_prompt_no_session():
  """无 session 时 prompt 为空"""
  sm = SessionManager()
  assert sm.to_prompt() == "", "无 session 应返回空字符串"
  print("  [PASS] 无 session prompt 为空")


def test_prompt_comment_session():
  """CommentSession prompt 包含核心要素"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个结局设计怎么样")], [])
  prompt = sm.to_prompt()
  assert "[聚焦话题]" in prompt
  assert "第1/3轮" in prompt
  assert "你觉得这个结局设计怎么样" in prompt
  assert "围绕这个话题" in prompt
  print("  [PASS] CommentSession prompt 正确")


def test_prompt_comment_session_with_history():
  """有历史记录时 prompt 包含之前讨论"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个结局设计怎么样")], [])
  sm.update_after_response("结局确实很精彩", [_c("对啊，结局设计太好了")])

  prompt = sm.to_prompt()
  assert "第2/3轮" in prompt
  assert "之前讨论" in prompt
  assert "结局确实很精彩" in prompt
  print("  [PASS] CommentSession prompt 含历史记录")


def test_prompt_video_session():
  """VideoSession prompt 包含核心要素"""
  sm = SessionManager()
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)
  prompt = sm.to_prompt()
  assert "[看视频模式]" in prompt
  assert "第1/5轮" in prompt
  assert "深入评论" in prompt
  print("  [PASS] VideoSession prompt 正确")


def test_prompt_video_session_interrupted():
  """VideoSession 被打断时 prompt 包含优先弹幕提示"""
  sm = SessionManager()
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)
  sm.on_comment_arrived([_c("主播看什么呢")])

  prompt = sm.to_prompt()
  assert "优先回复弹幕" in prompt, f"应含优先弹幕提示: {prompt}"
  print("  [PASS] VideoSession 被打断 prompt 含优先弹幕提示")


def test_prompt_video_session_with_history():
  """VideoSession 有历史时 prompt 提示不要重复"""
  sm = SessionManager()
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)
  sm.update_after_response("这段画面好震撼", [])

  prompt = sm.to_prompt()
  assert "你之前说了" in prompt
  assert "不要重复" in prompt
  print("  [PASS] VideoSession 含历史 prompt 正确")


# ──────────────────────────────────────────────
# 5. 相关性检测
# ──────────────────────────────────────────────

def test_relevance_keyword_overlap():
  """关键词重叠判定"""
  sm = SessionManager()
  anchor = "你觉得这个游戏的boss设计怎么样"
  # "boss设计" 中的 bi-gram 在弹幕中出现多次
  assert sm._has_relevant_comment([_c("boss设计确实厉害")], anchor)
  print("  [PASS] 关键词重叠判定正确")


def test_relevance_char_overlap():
  """字符重叠率判定"""
  sm = SessionManager()
  anchor = "你觉得这个游戏的boss设计怎么样"
  # "游戏设计" 与 anchor 有较多字符重叠
  assert sm._has_relevant_comment([_c("这个游戏设计")], anchor)
  print("  [PASS] 字符重叠率判定正确")


def test_relevance_noise_ignored():
  """噪声弹幕不算相关"""
  sm = SessionManager()
  anchor = "你觉得这个游戏的boss设计怎么样"
  assert not sm._has_relevant_comment([_c("哈哈哈")], anchor)
  assert not sm._has_relevant_comment([_c("666")], anchor)
  print("  [PASS] 噪声弹幕不算相关")


def test_relevance_empty():
  """空弹幕列表不相关"""
  sm = SessionManager()
  assert not sm._has_relevant_comment([], "任何话题")
  print("  [PASS] 空弹幕列表不相关")


def test_relevance_non_danmaku_ignored():
  """非弹幕事件不参与相关性检测"""
  sm = SessionManager()
  anchor = "你觉得这个游戏的boss设计怎么样"
  gift = _c("boss设计", event_type=EventType.GIFT)
  assert not sm._has_relevant_comment([gift], anchor)
  print("  [PASS] 非弹幕事件不参与相关性检测")


# ──────────────────────────────────────────────
# 6. 超时结束
# ──────────────────────────────────────────────

def test_session_timeout():
  """超时自动结束（模拟 last_active_at 很久以前）"""
  config = SessionConfig(relevance_timeout=1.0, comment_session_max_rounds=10)
  sm = SessionManager(config=config)
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])

  # 手动把 last_active_at 设到过去
  sm._session.last_active_at = time.monotonic() - 2.0

  # update 会检查 _should_end
  sm.update_after_response("回复", [])
  assert not sm.is_active, "超时后应结束"
  print("  [PASS] 超时自动结束")


# ──────────────────────────────────────────────
# 7. 配置禁用
# ──────────────────────────────────────────────

def test_disabled_config():
  """配置禁用时不开启任何 session"""
  config = SessionConfig(enabled=False)
  sm = SessionManager(config=config)

  result = sm.evaluate_new_session([_sc("加油", 100)], [])
  assert result is False, "禁用后不应开启"
  assert not sm.is_active

  result = sm.evaluate_new_session(
    [], [], has_scene_change=True, silence_seconds=100,
  )
  assert result is False, "禁用后不应开启 VideoSession"
  print("  [PASS] 配置禁用生效")


# ──────────────────────────────────────────────
# 8. debug_state
# ──────────────────────────────────────────────

def test_debug_state_inactive():
  """无 session 时 debug_state 正确"""
  sm = SessionManager()
  state = sm.debug_state()
  assert state["active"] is False
  assert "type" not in state
  print("  [PASS] debug_state 无 session 正确")


def test_debug_state_active():
  """有 session 时 debug_state 包含完整信息"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])
  sm.update_after_response("回复内容", [_c("说得好")])

  state = sm.debug_state()
  assert state["active"] is True
  assert state["type"] == "comment"
  assert state["round"] == 1
  assert state["max_rounds"] == 3
  assert "anchor_text" in state
  assert "history" in state
  assert isinstance(state["age_seconds"], (int, float))
  print("  [PASS] debug_state 有 session 信息完整")


# ──────────────────────────────────────────────
# 9. 手动结束
# ──────────────────────────────────────────────

def test_manual_end():
  """手动调用 end_session"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])
  assert sm.is_active

  sm.end_session()
  assert not sm.is_active
  assert sm.session is None
  print("  [PASS] 手动 end_session 正确")


def test_end_session_when_no_session():
  """无 session 时 end_session 不报错"""
  sm = SessionManager()
  sm.end_session()  # 不应抛异常
  assert not sm.is_active
  print("  [PASS] 无 session 时 end_session 安全")


# ──────────────────────────────────────────────
# 10. 边界情况
# ──────────────────────────────────────────────

def test_history_max_5():
  """历史记录最多保留 5 条"""
  config = SessionConfig(comment_session_max_rounds=10, stale_rounds_to_end=100)
  sm = SessionManager(config=config)
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])

  for i in range(8):
    sm.update_after_response(f"第{i}轮回复内容boss设计", [_c(f"boss设计第{i}轮")])

  assert len(sm.session.history) <= 5, f"历史记录应<=5，实际: {len(sm.session.history)}"
  print("  [PASS] 历史记录上限 5 条")


def test_expression_tag_stripped_from_history():
  """表情标签在历史记录中被去除"""
  sm = SessionManager()
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])
  sm.update_after_response(
    "#[wave][happy][joy] 这个boss超帅的 / このボスは超かっこいい",
    [_c("boss设计确实棒")],
  )
  history = sm.session.history[0]
  assert "#[" not in history, f"标签未去除: {history}"
  assert " / " not in history, f"日语翻译未去除: {history}"
  assert "这个boss超帅的" in history, f"中文内容应保留: {history}"
  print("  [PASS] 表情标签+日语翻译从历史记录去除")


def test_update_after_response_noop_when_no_session():
  """无 session 时 update_after_response 不报错"""
  sm = SessionManager()
  sm.update_after_response("任何回复", [_c("任何弹幕")])
  assert not sm.is_active
  print("  [PASS] 无 session 时 update_after_response 安全")


def test_video_session_stays_active_per_round():
  """VideoSession 每轮只要还在就重置 last_active_at"""
  config = SessionConfig(video_session_max_rounds=5)
  sm = SessionManager(config=config)
  sm.evaluate_new_session([], [], has_scene_change=True, silence_seconds=20.0)

  sm.update_after_response("画面描述1", [])
  assert sm.is_active
  assert sm.session.stale_rounds == 0, "VideoSession 应保持 stale=0"
  assert sm.session.round_count == 1
  print("  [PASS] VideoSession 每轮保持活跃")


def test_complete_lifecycle():
  """完整生命周期：开启 → 多轮 → 自然结束"""
  config = SessionConfig(comment_session_max_rounds=3, stale_rounds_to_end=3)
  sm = SessionManager(config=config)

  # 开启
  sm.evaluate_new_session([_c("你觉得这个游戏的boss设计怎么样")], [])
  assert sm.is_active and sm.session_type == SessionType.COMMENT

  # 第1轮：有相关
  sm.update_after_response("boss设计很精彩", [_c("boss设计确实棒")])
  assert sm.is_active and sm.session.round_count == 1

  # 第2轮：有相关
  sm.update_after_response("尤其是最终boss", [_c("最终boss太难了")])
  assert sm.is_active and sm.session.round_count == 2

  # 第3轮：达到最大轮数 → 结束
  sm.update_after_response("以后会更难的", [_c("boss设计加油")])
  assert not sm.is_active, "第3轮后应结束"
  print("  [PASS] 完整生命周期正确")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def main():
  tests = [
    # 1. 质量判定
    ("高质量弹幕判定", [
      test_quality_sc_always_high,
      test_quality_priority_always_high,
      test_quality_noise_excluded,
      test_quality_repetitive_excluded,
      test_quality_question,
      test_quality_opinion,
      test_quality_long_content,
      test_quality_non_danmaku_excluded,
      test_find_quality_comment,
      test_find_quality_comment_empty,
    ]),
    # 2. CommentSession
    ("CommentSession 生命周期", [
      test_comment_session_open,
      test_comment_session_round_increment,
      test_comment_session_end_max_rounds,
      test_comment_session_end_stale,
      test_comment_session_stale_reset,
      test_comment_session_no_duplicate_open,
    ]),
    # 3. VideoSession
    ("VideoSession 生命周期", [
      test_video_session_open,
      test_video_session_not_open_with_comments,
      test_video_session_not_open_insufficient_silence,
      test_video_session_interrupt_by_normal_comment,
      test_video_session_interrupt_cleared_when_no_comments,
      test_video_session_switch_to_comment_by_quality,
      test_video_session_end_max_rounds,
      test_on_comment_arrived_noop_for_comment_session,
    ]),
    # 4. Prompt 生成
    ("Prompt 生成", [
      test_prompt_no_session,
      test_prompt_comment_session,
      test_prompt_comment_session_with_history,
      test_prompt_video_session,
      test_prompt_video_session_interrupted,
      test_prompt_video_session_with_history,
    ]),
    # 5. 相关性
    ("相关性检测", [
      test_relevance_keyword_overlap,
      test_relevance_char_overlap,
      test_relevance_noise_ignored,
      test_relevance_empty,
      test_relevance_non_danmaku_ignored,
    ]),
    # 6. 超时
    ("超时与配置", [
      test_session_timeout,
      test_disabled_config,
    ]),
    # 7. debug_state
    ("调试状态", [
      test_debug_state_inactive,
      test_debug_state_active,
    ]),
    # 8. 手动结束
    ("手动结束", [
      test_manual_end,
      test_end_session_when_no_session,
    ]),
    # 9. 边界情况
    ("边界情况与完整流程", [
      test_history_max_5,
      test_expression_tag_stripped_from_history,
      test_update_after_response_noop_when_no_session,
      test_video_session_stays_active_per_round,
      test_complete_lifecycle,
    ]),
  ]

  total = 0
  passed = 0
  failed = 0

  for group_name, group_tests in tests:
    print(f"\n{'='*50}")
    print(f"  {group_name}")
    print(f"{'='*50}")
    for test_fn in group_tests:
      total += 1
      try:
        test_fn()
        passed += 1
      except AssertionError as e:
        failed += 1
        print(f"  [FAIL] {test_fn.__name__}: {e}")
      except Exception as e:
        failed += 1
        print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")

  print(f"\n{'='*50}")
  print(f"  结果: {passed}/{total} 通过, {failed} 失败")
  print(f"{'='*50}")
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  exit(main())
