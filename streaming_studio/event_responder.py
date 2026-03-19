"""
事件模板回复模块
将 GUARD_BUY / GIFT / ENTRY 事件从 LLM 管线中分离，使用模板回复替代 API 调用。

三级调度：
  高优先（独立回复）：GUARD_BUY > medium/large GIFT ≈ VIP入场 — 抢占弹幕
  LLM 管线：         DANMAKU / SUPER_CHAT
  前缀模式：         free/small GIFT + 普通 ENTRY — 附带在 LLM 回复前播报

设计要点：
- 高优先事件（GUARD / 贵礼物 / VIP入场）生成独立 StreamerResponse
- 低优先事件（便宜礼物 / 普通入场）不独立回复，
  通过 consume_prefix() 生成前缀 StreamerResponse 附带在 LLM 回复之前
- 前缀有独立冷却（prefix_cooldown），避免每条回复都带问候
- 无 LLM 回复时低优先事件通过 TTL 自然过期丢弃
"""

import json
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .config import EventResponderConfig
from .models import Comment, StreamerResponse, EventType, GUARD_LEVEL_NAMES

_GIFT_TIER_NAMES = ("free", "small", "medium", "large")
_DEFAULT_TEMPLATE_VOICE_EMOTION = "serenity"

_DEFAULT_GUARD_TEMPLATES = {
  "captain": ["#[wave][happy][joy] 谢谢{nickname}开通舰长！ / {nickname}、艦長ありがとう！"],
  "admiral": ["#[wave][happy][joy] 谢谢{nickname}开通提督！ / {nickname}、提督ありがとう！"],
  "governor": ["#[wave][happy][joy] 谢谢{nickname}开通总督！ / {nickname}、総督ありがとう！"],
}

_DEFAULT_GIFT_TEMPLATES = {
  "free": {"single": ["#[wave][happy][joy] 谢谢{nickname}的{gift_name}！ / {nickname}、ありがとう！"]},
  "small": {"single": ["#[wave][happy][joy] 谢谢{nickname}的{gift_name}！ / {nickname}、ありがとう！"]},
  "medium": {"single": ["#[wave][happy][joy] 谢谢{nickname}的{gift_name}！太感谢了！ / {nickname}、ありがとう！"]},
  "large": {"single": ["#[wave][happy][joy] 哇！谢谢{nickname}的{gift_name}！ / わぁ！{nickname}、ありがとう！"]},
}

_DEFAULT_ENTRY_TEMPLATES = {
  "single": ["#[wave][happy][joy] 欢迎{nickname}~ / {nickname}いらっしゃい〜"],
  "batch": ["#[wave][happy][joy] 欢迎{names}~ / {names}いらっしゃい〜"],
  "crowd": ["#[wave][happy][joy] 欢迎大家~ / みんないらっしゃい〜"],
  "vip": ["#[wave][happy][joy] 哇！{level}{nickname}来了！ / わぁ！{level}の{nickname}が来た！"],
}

_GUARD_LEVEL_TO_TIER = {1: "captain", 2: "admiral", 3: "governor"}
_TEMPLATE_TAG_RE = re.compile(r"(#\[[^\]]*\]\[[^\]]*\](?:\[[^\]]*\])?)")
_TEMPLATE_TAG_PARSE_RE = re.compile(r"#\[([^\]]*)\]\[([^\]]*)\](?:\[([^\]]*)\])?")


def _strip_japanese_tail(text: str) -> str:
  """从模板文本中剥离 ` / ` 后的日语部分，保留原始中文与空白。"""
  sep_idx = text.find(" / ")
  return text[:sep_idx] if sep_idx >= 0 else text


def _default_voice_emotion_for_template(expression: str) -> str:
  emotion = expression.strip().lower()
  mapping = {
    "happy": "joy",
    "excited": "anticipation",
    "smile": "serenity",
    "friendly": "serenity",
    "thoughtful": "serenity",
    "sad": "sadness",
    "sorry": "sadness",
    "surprised": "surprise",
    "angry": "anger",
    "annoyed": "agitation",
    "frustrated": "agitation",
    "curious": "curiosity",
    "shy": "shyness",
    "embarrassed": "shyness",
    "脸红": "shyness",
    "生气": "anger",
    "星星": "curiosity",
    "爱心": "joy",
    "脸黑": "disgust",
  }
  return mapping.get(emotion, _DEFAULT_TEMPLATE_VOICE_EMOTION)


def _normalize_template_tag(tag: str) -> str:
  matched = _TEMPLATE_TAG_PARSE_RE.fullmatch(tag.strip())
  if not matched:
    return tag
  motion = matched.group(1).strip()
  expression = matched.group(2).strip()
  voice_emotion = (matched.group(3) or "").strip()
  if not voice_emotion:
    voice_emotion = _default_voice_emotion_for_template(expression)
  return f"#[{motion}][{expression}][{voice_emotion}]"


def _normalize_template_text(text: str) -> str:
  """将双语模板归一化为中文主文本，保留三标签。"""
  if not text:
    return text

  parts = _TEMPLATE_TAG_RE.split(text)
  if len(parts) == 1:
    return _strip_japanese_tail(text).strip()

  rebuilt: list[str] = []
  leading = _strip_japanese_tail(parts[0]).strip()
  if leading:
    rebuilt.append(leading)

  i = 1
  while i + 1 < len(parts):
    tag = _normalize_template_tag(parts[i])
    body = _strip_japanese_tail(parts[i + 1]).rstrip()
    rebuilt.append(f"{tag}{body}")
    i += 2

  return "".join(rebuilt).strip()


class EventTemplateResponder:
  """
  事件模板回复器

  管理 GUARD_BUY / GIFT / ENTRY 三类事件队列，
  按优先级调度生成模板回复。
  """

  def __init__(
    self,
    persona: str,
    config: EventResponderConfig = EventResponderConfig(),
  ):
    self._config = config
    self._persona = persona

    # 高优先队列（独立回复）
    self._guard_queue: list[tuple[float, Comment]] = []
    self._gift_queue: list[tuple[float, Comment]] = []  # medium/large 礼物
    self._vip_entry_queue: list[tuple[float, Comment, str]] = []  # (time, comment, level_name)

    # 低优先队列（前缀模式）
    self._cheap_gift_queue: list[tuple[float, Comment]] = []  # free/small 礼物
    self._entry_queue: list[tuple[float, Comment]] = []  # 普通入场

    # 各类事件独立冷却
    self._last_guard_time: float = 0.0
    self._last_gift_time: float = 0.0
    self._last_vip_entry_time: float = 0.0
    self._last_prefix_time: float = 0.0

    # 模板去重（每类独立）
    self._recent_indices: dict[str, deque[int]] = {}

    # 加载模板
    self._guard_templates = self._load_json(persona, "guard_thanks.json", _DEFAULT_GUARD_TEMPLATES)
    self._gift_templates = self._load_json(persona, "gift_thanks.json", _DEFAULT_GIFT_TEMPLATES)
    self._entry_templates = self._load_json(persona, "entry_greetings.json", _DEFAULT_ENTRY_TEMPLATES)

  # ── 公开接口 ──

  def add_event(self, comment: Comment, guard_roster=None) -> None:
    """按 event_type 路由到对应队列

    礼物按价格分流：>= prefix_gift_threshold 走高优先独立回复，
    低于阈值走低优先前缀模式。
    """
    now = time.monotonic()

    if comment.event_type == EventType.GUARD_BUY:
      self._guard_queue.append((now, comment))

    elif comment.event_type == EventType.GIFT:
      if comment.price >= self._config.prefix_gift_threshold:
        self._gift_queue.append((now, comment))
      else:
        self._cheap_gift_queue.append((now, comment))

    elif comment.event_type == EventType.ENTRY:
      level_name = ""
      if guard_roster is not None:
        level_name = guard_roster.get_level_name(comment.user_id)
      if level_name:
        self._vip_entry_queue.append((now, comment, level_name))
      else:
        self._entry_queue.append((now, comment))

  def next_high_priority(
    self,
    allow_vip_entry: bool = True,
  ) -> Optional[StreamerResponse]:
    """取出最高优先级的 GUARD/GIFT/VIP入场 事件（优先级高于弹幕）

    allow_vip_entry=False 时，保留 VIP 入场队列，避免打断弹幕会话。
    """
    self._expire_stale()

    if self._guard_queue:
      return self._build_guard_response()

    if self._gift_queue and self._cooldown_ready("gift"):
      return self._build_gift_response()

    if allow_vip_entry and self._vip_entry_queue:
      return self._build_vip_entry_response()

    return None

  def consume_prefix(
    self,
    include_entry: bool = True,
  ) -> Optional[StreamerResponse]:
    """消费低优先事件生成前缀回复（附带在 LLM 回复之前）

    合并 cheap_gift + normal_entry 为一条短消息。
    有独立冷却控制（prefix_cooldown），避免每条 LLM 回复都带前缀。
    无待处理事件或冷却中返回 None。
    """
    self._expire_stale()
    if not self._cooldown_ready("prefix"):
      return None
    if not self._cheap_gift_queue and not (include_entry and self._entry_queue):
      return None

    parts = []
    reply_ids: list[str] = []

    if self._cheap_gift_queue:
      text, ids = self._build_cheap_gift_text()
      parts.append(text)
      reply_ids.extend(ids)

    if include_entry and self._entry_queue:
      text, ids = self._build_entry_text()
      parts.append(text)
      reply_ids.extend(ids)

    self._last_prefix_time = time.monotonic()
    combined = _normalize_template_text(" ".join(parts))
    return StreamerResponse(
      content=combined,
      reply_to=tuple(reply_ids),
      response_style="prefix_greet",
    )

  def has_pending(self) -> bool:
    return bool(self._guard_queue or self._gift_queue
                or self._cheap_gift_queue
                or self._entry_queue or self._vip_entry_queue)

  # ── 模板构建 ──

  def _build_guard_response(self) -> StreamerResponse:
    """上舰感谢：每条单独回复，按等级选档，拼接多条模板"""
    _, comment = self._guard_queue.pop(0)
    self._last_guard_time = time.monotonic()

    tier = _GUARD_LEVEL_TO_TIER.get(comment.guard_level, "captain")
    level = GUARD_LEVEL_NAMES.get(comment.guard_level, "舰长")
    pool = self._guard_templates.get(tier, [])
    n = self._config.guard_min_sentences
    templates = self._pick_multiple_templates(f"guard_{tier}", pool, n)
    text = _normalize_template_text(" ".join(
      t.format(nickname=comment.nickname, level=level) for t in templates
    ))

    return StreamerResponse(
      content=text,
      reply_to=(comment.id,),
      response_style="guard_thanks",
    )

  def _build_gift_response(self) -> StreamerResponse:
    """礼物感谢：同档位可批量合并"""
    self._last_gift_time = time.monotonic()

    # 按最高档位分组
    best_tier_idx = 0
    for _, c in self._gift_queue:
      best_tier_idx = max(best_tier_idx, self._get_gift_tier_index(c.price))

    tier_name = _GIFT_TIER_NAMES[best_tier_idx]
    tier_templates = self._gift_templates.get(tier_name, {})

    # 取出同档位或更高的礼物
    to_respond = []
    remaining = []
    for item in self._gift_queue:
      _, c = item
      if self._get_gift_tier_index(c.price) >= best_tier_idx:
        to_respond.append(item)
      else:
        remaining.append(item)
    self._gift_queue = remaining

    if not to_respond:
      return None

    comments = [c for _, c in to_respond]

    # 去重昵称
    nicknames = []
    seen = set()
    for c in comments:
      if c.nickname not in seen:
        nicknames.append(c.nickname)
        seen.add(c.nickname)

    # 根据档位决定感谢句数
    sentence_map = {
      "large": self._config.large_gift_min_sentences,
      "medium": self._config.medium_gift_min_sentences,
    }
    n_sentences = sentence_map.get(tier_name, 1)

    if len(comments) == 1:
      c = comments[0]
      pool = tier_templates.get("single", [])
      num_str = f"x{c.gift_num}" if c.gift_num > 1 else ""
      fmt_kwargs = dict(
        nickname=c.nickname,
        gift_name=c.gift_name or "礼物",
        gift_num=c.gift_num,
        num_str=num_str,
      )
      if n_sentences > 1:
        templates = self._pick_multiple_templates(f"gift_{tier_name}_single", pool, n_sentences)
        text = _normalize_template_text(" ".join(t.format(**fmt_kwargs) for t in templates))
      else:
        template = self._pick_template(f"gift_{tier_name}_single", pool)
        text = _normalize_template_text(template.format(**fmt_kwargs))
    else:
      pool = tier_templates.get("batch", tier_templates.get("single", []))
      template = self._pick_template(f"gift_{tier_name}_batch", pool)
      names = self._format_names(nicknames, max_n=self._config.entry_max_names)
      text = _normalize_template_text(template.format(
        names=names,
        nickname=nicknames[0],
        gift_name=comments[0].gift_name or "礼物",
        gift_num=sum(c.gift_num for c in comments),
        num_str="",
      ))

    reply_to = tuple(c.id for c in comments)
    return StreamerResponse(
      content=text,
      reply_to=reply_to,
      response_style="gift_thanks",
    )

  def _build_vip_entry_response(self) -> StreamerResponse:
    """VIP 入场：舰长/提督/总督进直播间，单条回复"""
    _, comment, level_name = self._vip_entry_queue.pop(0)
    self._last_vip_entry_time = time.monotonic()

    pool = self._entry_templates.get("vip", [])
    template = self._pick_template("entry_vip", pool)
    text = _normalize_template_text(
      template.format(nickname=comment.nickname, level=level_name)
    )

    return StreamerResponse(
      content=text,
      reply_to=(comment.id,),
      response_style="vip_entry",
    )

  def _build_cheap_gift_text(self) -> tuple[str, list[str]]:
    """消费 cheap_gift_queue，返回 (模板文本, comment_id 列表)"""
    items = self._cheap_gift_queue
    self._cheap_gift_queue = []

    comments = [c for _, c in items]
    nicknames = []
    seen = set()
    for c in comments:
      if c.nickname not in seen:
        nicknames.append(c.nickname)
        seen.add(c.nickname)

    # 选取最高档位的模板（free/small 范围内）
    best_tier_idx = 0
    for c in comments:
      best_tier_idx = max(best_tier_idx, self._get_gift_tier_index(c.price))
    tier_name = _GIFT_TIER_NAMES[best_tier_idx]
    tier_templates = self._gift_templates.get(tier_name, {})

    if len(comments) == 1:
      c = comments[0]
      pool = tier_templates.get("single", [])
      template = self._pick_template(f"gift_{tier_name}_single", pool)
      num_str = f"x{c.gift_num}" if c.gift_num > 1 else ""
      text = _normalize_template_text(template.format(
        nickname=c.nickname,
        gift_name=c.gift_name or "礼物",
        gift_num=c.gift_num,
        num_str=num_str,
      ))
    else:
      pool = tier_templates.get("batch", tier_templates.get("single", []))
      template = self._pick_template(f"gift_{tier_name}_batch", pool)
      names = self._format_names(nicknames, max_n=self._config.entry_max_names)
      text = _normalize_template_text(template.format(
        names=names,
        nickname=nicknames[0],
        gift_name=comments[0].gift_name or "礼物",
        gift_num=sum(c.gift_num for c in comments),
        num_str="",
      ))

    return text, [c.id for c in comments]

  def _build_entry_text(self) -> tuple[str, list[str]]:
    """消费 entry_queue，返回 (模板文本, comment_id 列表)"""
    entries = self._entry_queue
    self._entry_queue = []

    comments = [c for _, c in entries]
    nicknames = []
    seen = set()
    for c in comments:
      if c.nickname not in seen:
        nicknames.append(c.nickname)
        seen.add(c.nickname)

    count = len(nicknames)

    if count == 1:
      category = "single"
      pool = self._entry_templates.get(category, [])
      template = self._pick_template(f"entry_{category}", pool)
      text = _normalize_template_text(template.format(nickname=nicknames[0]))
    elif count < self._config.entry_crowd_threshold:
      category = "batch"
      pool = self._entry_templates.get(category, [])
      template = self._pick_template(f"entry_{category}", pool)
      names = self._format_names(nicknames, max_n=self._config.entry_max_names)
      text = _normalize_template_text(template.format(names=names))
    else:
      category = "crowd"
      pool = self._entry_templates.get(category, [])
      template = self._pick_template(f"entry_{category}", pool)
      text = _normalize_template_text(template)

    return text, [c.id for c in comments]

  # ── 工具方法 ──

  def _get_gift_tier_index(self, price: float) -> int:
    thresholds = self._config.gift_tier_thresholds
    tier = 0
    for i, t in enumerate(thresholds):
      if price >= t:
        tier = i
    return tier

  def _cooldown_ready(self, event_type: str) -> bool:
    now = time.monotonic()
    if event_type == "guard":
      return (now - self._last_guard_time) >= self._config.guard_cooldown
    if event_type == "gift":
      return (now - self._last_gift_time) >= self._config.gift_cooldown
    if event_type == "prefix":
      return (now - self._last_prefix_time) >= self._config.prefix_cooldown
    return True

  def _expire_stale(self) -> None:
    now = time.monotonic()
    gift_ttl = self._config.gift_ttl
    entry_ttl = self._config.entry_ttl
    # GUARD_BUY 永不过期
    self._gift_queue = [(t, c) for t, c in self._gift_queue if now - t < gift_ttl]
    self._cheap_gift_queue = [(t, c) for t, c in self._cheap_gift_queue if now - t < gift_ttl]
    self._entry_queue = [(t, c) for t, c in self._entry_queue if now - t < entry_ttl]
    # VIP 入场也不过期（舰长进来必须欢迎）

  def _format_names(self, nicknames: list[str], max_n: int = 3) -> str:
    if len(nicknames) <= max_n:
      return "、".join(nicknames)
    shown = "、".join(nicknames[:max_n - 1])
    return f"{shown}等{len(nicknames)}位朋友"

  def _pick_multiple_templates(self, category_key: str, pool: list[str], n: int) -> list[str]:
    """从模板池中选取 n 条不重复模板"""
    results = []
    for _ in range(min(n, len(pool))):
      t = self._pick_template(category_key, pool)
      results.append(t)
    return results if results else [self._pick_template(category_key, pool)]

  def _pick_template(self, category_key: str, pool: list[str]) -> str:
    if not pool:
      return "#[wave][happy] 谢谢！"

    if category_key not in self._recent_indices:
      self._recent_indices[category_key] = deque(maxlen=self._config.recent_dedup_count)

    recent = self._recent_indices[category_key]
    candidates = [i for i in range(len(pool)) if i not in recent]
    if not candidates:
      recent.clear()
      candidates = list(range(len(pool)))

    idx = random.choice(candidates)
    recent.append(idx)
    return pool[idx]

  def _load_json(self, persona: str, filename: str, defaults: dict) -> dict:
    path = Path("personas") / persona / filename
    if path.exists():
      try:
        with open(path, "r", encoding="utf-8") as f:
          return json.load(f)
      except (json.JSONDecodeError, KeyError) as e:
        print(f"[事件回复] 模板文件解析失败 ({path}): {e}，使用默认模板")
    return dict(defaults)

  def debug_state(self) -> dict:
    return {
      "enabled": self._config.enabled,
      "guard_pending": len(self._guard_queue),
      "gift_pending": len(self._gift_queue),
      "cheap_gift_pending": len(self._cheap_gift_queue),
      "entry_pending": len(self._entry_queue),
      "vip_entry_pending": len(self._vip_entry_queue),
    }
