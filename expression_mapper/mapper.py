"""
表情动作语义映射器

将 LLM 输出的自由文本 #[action][emotion] 标签映射到
expression_motion_mapping.json 中的固定集合。

采用 sentence embedding + cosine similarity 实现语义最近邻匹配，
预计算目标集合的向量，运行时仅需 embed 查询词并做点积，延迟极低。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(r"#\[([^\]]*)\]\[([^\]]*)\]")

_EMOTION_KEYWORDS: dict[str, list[str]] = {
  "脸红": ["blush", "embarrassed", "shy", "flustered", "tsundere", "nervous"],
  "生气": ["angry", "mad", "annoyed", "frustrated", "furious", "rage"],
  "哭": ["cry", "sad", "upset", "whine", "tearful", "sobbing", "disappointed"],
  "- -": ["deadpan", "unamused", "bored", "blank", "poker face", "indifferent", "cold"],
  "口罩": ["mask", "hiding", "cover face", "extremely shy", "facepalm", "nervous", "scared"],
  "吐舌": ["tongue out", "playful", "teasing", "mischievous", "cheeky", "smug", "proud", "satisfied", "得意"],
  "外套": ["coat", "caring", "warm", "cold weather", "concern", "friendly"],
  "星星": ["sparkle", "starry eyes", "interested", "excited", "happy", "amazed", "开心", "快乐"],
  "比心": ["heart", "love", "sincere", "touched", "affection", "heartfelt"],
  "水印": ["watermark", "special", "neutral"],
  "眼镜": ["glasses", "serious", "analytical", "lecturing", "smart", "thoughtful", "thinking", "沉思"],
  "脸黑": ["dark face", "speechless", "cringe", "awkward", "fail", "facepalm"],
  "荷包蛋": ["fried egg", "mind blown", "exhausted", "done", "broken", "overwhelmed", "shocked"],
  "阿尼亚": ["anya", "curious", "cute", "breaking character", "surprised", "waku waku"],
}

_MOTION_KEYWORDS: dict[str, list[str]] = {
  "Idle": ["idle", "stand", "still", "default", "waiting", "calm", "relaxed"],
  "Shake": ["shake", "shake head", "deny", "disagree", "refuse", "no"],
  "Nod": ["nod", "agree", "acknowledge", "yes", "listen", "understand"],
  "Jump": ["jump", "hop", "happy", "excited", "celebrate", "cheer", "laugh"],
  "Sulk": ["sulk", "droop", "sad", "depressed", "down", "disappointed", "sigh"],
  "Wave": ["wave", "hello", "goodbye", "greeting", "hi", "bye", "welcome"],
  "Stomp": ["stomp", "angry", "frustrated", "tantrum", "furious", "rage"],
  "Spin": ["spin", "twirl", "very happy", "ecstatic", "overjoyed", "dance"],
  "Peek": ["peek", "peep", "curious", "sneaky", "watching", "spy", "shy look"],
  "Lean": ["lean", "lean forward", "interested", "closer", "intrigued", "attentive"],
}


_EN_ZH_EMOTION: dict[str, str] = {
  "happy": "开心 快乐 高兴 眼前一亮",
  "excited": "兴奋 激动 期待 眼前一亮",
  "friendly": "友好 温暖 亲切 关心",
  "thoughtful": "沉思 认真 思考 分析 严肃 讲道理",
  "sad": "难过 悲伤 伤心 委屈",
  "angry": "生气 愤怒 恼火 发火",
  "sorry": "抱歉 道歉 委屈",
  "shy": "害羞 脸红 不好意思 被拆穿",
  "embarrassed": "尴尬 害臊 脸红 被拆穿",
  "proud": "得意 骄傲 自豪 嘚瑟 调皮 吐舌头",
  "curious": "好奇 感兴趣 想知道 眼前一亮",
  "bored": "无聊 无语 嫌弃 懒得理",
  "annoyed": "烦躁 不耐烦 嫌弃 无语 懒得理",
  "surprised": "惊讶 意外 吃惊 眼前一亮",
  "confused": "困惑 迷茫 不明白 无话可说",
  "determined": "坚定 认真 不服输 好胜",
  "playful": "调皮 搞怪 玩闹 吐舌头 得意",
  "smug": "得意 嘚瑟 沾沾自喜 调皮 吐舌头",
  "nervous": "紧张 忐忑 慌张 害羞 脸红",
  "touched": "感动 真情流露 比心",
  "love": "喜欢 心动 比心 真情流露",
  "worry": "担心 担忧 忧虑 关心",
  "flustered": "手足无措 慌乱 脸红 害羞",
  "cold": "冷淡 高冷 不想理你 嫌弃 无语",
  "speechless": "无语 无话可说 尴尬 脸黑 翻车",
}

_EN_ZH_MOTION: dict[str, str] = {
  "laugh": "大笑 哈哈笑 开心跳",
  "smile": "微笑 温和 平静 站着 待机",
  "nod": "点头 认同 同意",
  "wave": "挥手 打招呼 再见",
  "clap": "鼓掌 拍手 赞赏",
  "shake head": "摇头 否认 不同意",
  "hands on hips": "叉腰 跺脚 生气 不服",
  "cheering up": "欢呼 跳跃 兴奋",
  "ok gesture": "点头 认同 没问题",
  "heart gesture": "比心 前倾 真情",
  "thinking": "待机 沉思 认真",
  "lean forward": "前倾 凑近 感兴趣",
  "peek": "偷看 探头 好奇",
  "sigh": "叹气 垂头 丧气",
  "dance": "转圈 跳舞 得意",
  "stomp": "跺脚 生气 不满",
  "spin": "转圈 高兴 得意忘形",
  "jump": "跳跃 开心 兴奋",
}


@dataclass
class MappedTag:
  """一个 #[action][emotion] 标签的映射结果"""
  original_action: str
  original_emotion: str
  mapped_motion: str
  mapped_expression: str
  motion_score: float
  expression_score: float


@dataclass
class MapperResult:
  """整句文本的映射结果"""
  mapped_text: str
  tags: list[MappedTag] = field(default_factory=list)


class ExpressionMotionMapper:
  """
  语义映射器：LLM 自由标签 → 固定表情/动作集

  初始化时从 mapping JSON 加载目标集合并预计算 embeddings，
  运行时用 cosine similarity 做最近邻查找。
  查询结果会缓存，相同词不会重复 embed。
  """

  def __init__(
    self,
    mapping_path: Path,
    embeddings=None,
  ) -> None:
    self._cache: dict[str, tuple[str, float]] = {}

    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    self._emotions: list[dict] = raw.get("emotions", [])
    self._motions: list[dict] = raw.get("motions", [])

    self._embeddings = embeddings
    self._emotion_names: list[str] = [e["name"] for e in self._emotions]
    self._motion_names: list[str] = [m["name"] for m in self._motions]

    self._emotion_vectors: Optional[np.ndarray] = None
    self._motion_vectors: Optional[np.ndarray] = None

    if embeddings is not None:
      self._precompute()

  def lazy_init(self, embeddings) -> None:
    """延迟初始化 embedding 模型（等 MemoryManager 就绪后调用）"""
    if self._emotion_vectors is not None:
      return
    self._embeddings = embeddings
    self._precompute()

  def _precompute(self) -> None:
    """预计算目标集合的 embedding 向量"""
    emotion_texts = []
    for e in self._emotions:
      kw = _EMOTION_KEYWORDS.get(e["name"], [])
      text = f"{e['name']} ({', '.join(kw)}): {e['desc']}" if kw else f"{e['name']}: {e['desc']}"
      emotion_texts.append(text)

    motion_texts = []
    for m in self._motions:
      kw = _MOTION_KEYWORDS.get(m["name"], [])
      text = f"{m['name']} ({', '.join(kw)}): {m['desc']}" if kw else f"{m['name']}: {m['desc']}"
      motion_texts.append(text)

    try:
      ev = self._embeddings.embed_documents(emotion_texts)
      mv = self._embeddings.embed_documents(motion_texts)
      self._emotion_vectors = self._normalize(np.array(ev, dtype=np.float32))
      self._motion_vectors = self._normalize(np.array(mv, dtype=np.float32))
      logger.info(
        "ExpressionMotionMapper 预计算完成: %d emotions, %d motions",
        len(self._emotion_names), len(self._motion_names),
      )
    except Exception as e:
      logger.error("ExpressionMotionMapper 预计算失败: %s", e)

  def map_response(self, text: str) -> MapperResult:
    """
    解析并映射整段回复文本中所有 #[action][emotion] 标签。

    Returns:
      MapperResult，包含替换后文本和每个标签的映射详情。
    """
    if self._emotion_vectors is None or self._motion_vectors is None:
      return MapperResult(mapped_text=text)

    tags: list[MappedTag] = []

    def _replace(match: re.Match) -> str:
      action_raw = match.group(1).strip()
      emotion_raw = match.group(2).strip()

      mapped_motion, m_score = self._find_nearest_motion(action_raw)
      mapped_expr, e_score = self._find_nearest_emotion(emotion_raw)

      tags.append(MappedTag(
        original_action=action_raw,
        original_emotion=emotion_raw,
        mapped_motion=mapped_motion,
        mapped_expression=mapped_expr,
        motion_score=round(m_score, 4),
        expression_score=round(e_score, 4),
      ))
      return f"#[{mapped_motion}][{mapped_expr}]"

    mapped_text = _TAG_PATTERN.sub(_replace, text)
    return MapperResult(mapped_text=mapped_text, tags=tags)

  def _find_nearest_emotion(self, query: str) -> tuple[str, float]:
    cache_key = f"e:{query}"
    if cache_key in self._cache:
      return self._cache[cache_key]

    enriched = self._enrich_query(query, _EN_ZH_EMOTION)
    result = self._nearest(enriched, self._emotion_names, self._emotion_vectors)
    self._cache[cache_key] = result
    return result

  def _find_nearest_motion(self, query: str) -> tuple[str, float]:
    cache_key = f"m:{query}"
    if cache_key in self._cache:
      return self._cache[cache_key]

    enriched = self._enrich_query(query, _EN_ZH_MOTION)
    result = self._nearest(enriched, self._motion_names, self._motion_vectors)
    self._cache[cache_key] = result
    return result

  @staticmethod
  def _enrich_query(query: str, lookup: dict[str, str]) -> str:
    """将英文查询增强为中英双语，提升中文 embedding 模型的匹配准确率"""
    key = query.lower().strip()
    zh = lookup.get(key)
    if zh:
      return f"{query} {zh}"
    for en_key, zh_val in lookup.items():
      if en_key in key or key in en_key:
        return f"{query} {zh_val}"
    return query

  def _nearest(
    self,
    query: str,
    names: list[str],
    vectors: np.ndarray,
  ) -> tuple[str, float]:
    """用 cosine similarity 找最近邻"""
    try:
      qv = np.array(
        self._embeddings.embed_query(query), dtype=np.float32,
      ).reshape(1, -1)
      qv = self._normalize(qv)
      scores = (qv @ vectors.T).flatten()
      idx = int(np.argmax(scores))
      return names[idx], float(scores[idx])
    except Exception as e:
      logger.error("语义匹配失败 (query=%s): %s", query, e)
      return names[0], 0.0

  @staticmethod
  def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms
