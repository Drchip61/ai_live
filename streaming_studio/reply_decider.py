"""
回复决策器 + 弹幕聚类器
两阶段判断主播是否应该回复：规则快筛（免费） + LLM 精判（Haiku）
两阶段弹幕聚类：循环节规则快筛（免费） + 语义 embedding 精判
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from .config import ReplyDeciderConfig, CommentClustererConfig
from .models import Comment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplyDecision:
  """回复决策结果"""
  should_reply: bool
  urgency: float
  reason: str
  phase: str  # "rule" or "llm"


class ReplyDecider:
  """
  两阶段回复决策器

  Phase 1（规则快筛，免费）：
    处理明确场景——必须回复（提问、高活跃）或建议跳过（纯反应词、刷屏）

  Phase 2（LLM 精判，Haiku）：
    当规则无法决定时，用轻量 LLM 综合弹幕内容、场景描述和沉默时长做判断
  """

  def __init__(
    self,
    config: Optional[ReplyDeciderConfig] = None,
    llm_model: Optional[BaseChatModel] = None,
    judge_prompt: str = "",
  ):
    self.config = config or ReplyDeciderConfig()
    self._llm = llm_model
    self._judge_prompt = judge_prompt
    self._skip_set = set(p.lower() for p in self.config.skip_patterns)

  def rule_check(
    self,
    old_comments: list[Comment],
    new_comments: list[Comment],
  ) -> Optional[ReplyDecision]:
    """
    Phase 1: 规则快筛

    Returns:
      明确决策时返回 ReplyDecision，不确定时返回 None
    """
    all_comments = old_comments + new_comments

    if not all_comments:
      return ReplyDecision(False, 0, "无弹幕", "rule")

    # 优先弹幕（手动输入）→ 必须回复
    if any(c.priority for c in all_comments):
      return ReplyDecision(True, 9, "有优先弹幕", "rule")

    # 新弹幕数量超过阈值 → 直接回复（聊天很活跃）
    if len(new_comments) >= self.config.must_reply_comment_count:
      return ReplyDecision(True, 8, f"新弹幕数量多({len(new_comments)}条)", "rule")

    # 包含提问（问号） → 必须回复
    for c in new_comments:
      if "?" in c.content or "？" in c.content:
        stripped = c.content.replace("?", "").replace("？", "").strip()
        if len(stripped) >= 2:
          return ReplyDecision(True, 8, f"观众提问: {c.content[:20]}", "rule")

    # 检查是否全是低质量内容
    low_quality_count = 0
    for c in new_comments:
      content = c.content.strip().lower()
      is_low = (
        len(content) <= self.config.min_quality_length
        or content in self._skip_set
        or self._is_repetitive(content)
      )
      if is_low:
        low_quality_count += 1

    if new_comments and low_quality_count == len(new_comments):
      return ReplyDecision(False, 1, "全部为低质量弹幕", "rule")

    # 不确定 → 交给 Phase 2
    return None

  async def llm_judge(
    self,
    old_comments: list[Comment],
    new_comments: list[Comment],
    scene_description: Optional[str] = None,
    silence_seconds: float = 0,
  ) -> ReplyDecision:
    """
    Phase 2: LLM 精判

    用轻量模型判断当前弹幕是否值得回复
    """
    if self._llm is None:
      return ReplyDecision(True, 5, "无LLM可用，默认回复", "llm")

    parts = []
    if scene_description:
      parts.append(f"[当前画面] {scene_description}")
    if silence_seconds > 0:
      parts.append(f"[沉默时长] 距上次回复已过 {int(silence_seconds)} 秒")

    if old_comments:
      lines = [f"  {c.nickname}: {c.content}" for c in old_comments[-5:]]
      parts.append("[旧弹幕]\n" + "\n".join(lines))

    if new_comments:
      lines = [f"  {c.nickname}: {c.content}" for c in new_comments]
      parts.append("[新弹幕]\n" + "\n".join(lines))

    user_text = "\n\n".join(parts)

    messages = [
      SystemMessage(content=self._judge_prompt),
      HumanMessage(content=user_text),
    ]

    try:
      result = await self._llm.ainvoke(messages)
      text = result.content if hasattr(result, "content") else str(result)
      return self._parse_judge_response(text)
    except Exception as e:
      logger.error("LLM 精判调用失败: %s", e)
      return ReplyDecision(False, 2, f"精判异常({e})，默认跳过", "llm")

  async def should_reply(
    self,
    old_comments: list[Comment],
    new_comments: list[Comment],
    scene_description: Optional[str] = None,
    last_reply_time: Optional[datetime] = None,
  ) -> ReplyDecision:
    """
    完整两阶段决策

    Args:
      old_comments: 上次回复前的弹幕
      new_comments: 新弹幕
      scene_description: 当前场景描述（VLM 模式）
      last_reply_time: 上次回复时间（用于计算沉默时长）
    """
    # Phase 1
    decision = self.rule_check(old_comments, new_comments)
    if decision is not None:
      logger.info("回复决策[规则]: %s (urgency=%.0f, %s)",
                  "回复" if decision.should_reply else "跳过",
                  decision.urgency, decision.reason)
      return decision

    # Phase 2
    silence = 0.0
    if last_reply_time:
      silence = (datetime.now() - last_reply_time).total_seconds()

    decision = await self.llm_judge(
      old_comments, new_comments, scene_description, silence,
    )
    logger.info("回复决策[LLM]: %s (urgency=%.0f, %s)",
                "回复" if decision.should_reply else "跳过",
                decision.urgency, decision.reason)
    return decision

  async def should_proactive_speak(
    self,
    prev_scene: Optional[str],
    current_scene: Optional[str],
    silence_seconds: float,
  ) -> ReplyDecision:
    """
    判断是否应该主动发言（无弹幕时，基于画面变化）

    Args:
      prev_scene: 上一次场景描述
      current_scene: 当前场景描述
      silence_seconds: 沉默时长（秒）
    """
    if silence_seconds < self.config.proactive_silence_threshold:
      return ReplyDecision(False, 0, "沉默时间不足", "rule")

    if not current_scene:
      return ReplyDecision(False, 0, "无场景信息", "rule")

    # 没有上一帧基线时，允许在沉默阈值后先主动开场一次
    if not prev_scene:
      return ReplyDecision(True, 6, "沉默已久，基于当前画面主动开场", "rule")

    if current_scene == prev_scene:
      return ReplyDecision(False, 0, "场景无变化", "rule")

    # 用 LLM 判断场景变化是否有意义
    if self._llm is None:
      return ReplyDecision(True, 5, "场景变化，默认发言", "rule")

    prompt = (
      f"[上一次画面] {prev_scene}\n"
      f"[当前画面] {current_scene}\n"
      f"[沉默时长] {int(silence_seconds)} 秒\n\n"
      f"画面是否发生了值得主播评论的重要变化？"
    )
    messages = [
      SystemMessage(content=self._judge_prompt),
      HumanMessage(content=prompt),
    ]
    try:
      result = await self._llm.ainvoke(messages)
      text = result.content if hasattr(result, "content") else str(result)
      decision = self._parse_judge_response(text)
      logger.info("主动发言决策: %s (urgency=%.0f, %s)",
                  "发言" if decision.should_reply else "沉默",
                  decision.urgency, decision.reason)
      return decision
    except Exception as e:
      logger.error("主动发言判断失败: %s", e)
      return ReplyDecision(False, 0, f"判断异常({e})", "llm")

  def _parse_judge_response(self, text: str) -> ReplyDecision:
    """解析 LLM 精判的 true/false 响应"""
    normalized = text.strip().lower()
    if normalized.startswith("true"):
      return ReplyDecision(True, 5, "LLM判断", "llm")
    if normalized.startswith("false"):
      return ReplyDecision(False, 1, "LLM判断", "llm")

    logger.warning("LLM 精判响应解析失败: %s", text[:100])
    return ReplyDecision(False, 2, "响应解析失败，默认跳过", "llm")

  @staticmethod
  def _is_repetitive(content: str) -> bool:
    """检查是否为重复字符（如 "哈哈哈哈"、"666666"）"""
    if len(content) <= 1:
      return True
    unique_chars = set(content)
    return len(unique_chars) <= 2


# ---------------------------------------------------------------------------
# 弹幕聚类器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommentCluster:
  """单个弹幕簇"""
  representative: Comment
  """代表弹幕（选最长内容的那条）"""

  members: tuple[Comment, ...] = field(default_factory=tuple)
  """所有成员（含代表）"""

  merge_reason: str = "pattern"
  """合并原因: "pattern"（循环节规则）或 "semantic"（语义相似）"""

  @property
  def count(self) -> int:
    return len(self.members)


@dataclass(frozen=True)
class ClusterResult:
  """弹幕聚类结果"""
  clusters: tuple[CommentCluster, ...] = field(default_factory=tuple)
  """聚类后的簇"""

  singles: tuple[Comment, ...] = field(default_factory=tuple)
  """未归入任何簇的独立弹幕"""

  def representatives(self) -> list[Comment]:
    """去重后的代表弹幕列表（单条 + 每簇代表），按时间排序"""
    result = list(self.singles)
    for cluster in self.clusters:
      result.append(cluster.representative)
    return sorted(result, key=lambda c: c.timestamp)

  def cluster_for(self, comment_id: str) -> Optional[CommentCluster]:
    """根据弹幕 ID 查找所属簇，不在任何簇中返回 None"""
    for cluster in self.clusters:
      if any(m.id == comment_id for m in cluster.members):
        return cluster
    return None


class CommentClusterer:
  """
  两阶段弹幕聚类器

  Phase 1（规则快筛，零成本）：
    提取每条弹幕的最小循环节，循环节相同的弹幕直接归组。
    "666"、"6666"、"66666" → 循环节 "6" → 同簇
    "哈哈"、"哈哈哈哈" → 循环节 "哈" → 同簇
    "233"、"233233" → 循环节 "233" → 同簇

  Phase 2（语义聚类，需 embeddings）：
    对规则阶段未归组的弹幕做 embedding + cosine similarity，
    超阈值的合并。无 embeddings 时自动跳过此阶段。
  """

  def __init__(
    self,
    config: Optional[CommentClustererConfig] = None,
    embeddings=None,
  ):
    """
    Args:
      config: 聚类器配置
      embeddings: HuggingFaceEmbeddings 实例（可选，传入后启用语义阶段）
    """
    self.config = config or CommentClustererConfig()
    self._embeddings = embeddings

  def cluster(self, comments: list[Comment]) -> ClusterResult:
    """
    完整两阶段聚类

    Args:
      comments: 待聚类的弹幕列表

    Returns:
      ClusterResult 包含簇列表和独立弹幕
    """
    if not comments:
      return ClusterResult()

    # priority 弹幕不参与聚类
    priority = [c for c in comments if c.priority]
    normal = [c for c in comments if not c.priority]

    if not normal:
      return ClusterResult(singles=tuple(priority))

    # Phase 1: 循环节规则快筛
    rule_clusters, remaining = self._rule_cluster(normal)

    # Phase 2: 语义聚类
    semantic_clusters, still_remaining = self._semantic_cluster(remaining)

    # 汇总
    all_clusters = rule_clusters + semantic_clusters
    all_singles = priority + still_remaining

    logger.info(
      "弹幕聚类: %d条 → %d簇 + %d条独立 (规则%d簇, 语义%d簇)",
      len(comments), len(all_clusters), len(all_singles),
      len(rule_clusters), len(semantic_clusters),
    )

    return ClusterResult(
      clusters=tuple(all_clusters),
      singles=tuple(all_singles),
    )

  def _rule_cluster(
    self,
    comments: list[Comment],
  ) -> tuple[list[CommentCluster], list[Comment]]:
    """
    Phase 1: 按循环节归组

    Returns:
      (成功归组的簇列表, 未归组的弹幕列表)
    """
    groups: dict[str, list[Comment]] = {}
    for c in comments:
      content = c.content.strip().lower()
      unit = self._extract_repeating_unit(content)

      # 循环节长度超过阈值的不视为循环模式，按原文分组
      if len(unit) > self.config.max_pattern_unit_length:
        key = content
      else:
        key = unit

      groups.setdefault(key, []).append(c)

    clusters = []
    ungrouped = []
    for members in groups.values():
      if len(members) >= self.config.min_cluster_size:
        rep = max(members, key=lambda c: len(c.content))
        clusters.append(CommentCluster(
          representative=rep,
          members=tuple(members),
          merge_reason="pattern",
        ))
      else:
        ungrouped.extend(members)

    return clusters, ungrouped

  def _semantic_cluster(
    self,
    comments: list[Comment],
  ) -> tuple[list[CommentCluster], list[Comment]]:
    """
    Phase 2: 语义聚类（增量阈值法）

    Returns:
      (成功归组的簇列表, 未归组的弹幕列表)
    """
    if not comments or self._embeddings is None:
      return [], list(comments) if comments else []

    # 太少不值得做 embedding
    if len(comments) < self.config.min_cluster_size:
      return [], list(comments)

    try:
      texts = [c.content for c in comments]
      vectors = self._embeddings.embed_documents(texts)
      matrix = np.array(vectors, dtype=np.float32)
    except Exception as e:
      logger.warning("语义聚类 embedding 失败: %s", e)
      return [], list(comments)

    # 归一化（bge 输出通常已归一化，但安全起见）
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    # 增量阈值聚类
    # cluster_indices[i] = 该簇包含的 comments 下标列表
    # centroids[i] = 该簇的质心向量
    cluster_indices: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for idx in range(len(comments)):
      vec = matrix[idx]
      best_sim = -1.0
      best_cluster = -1

      for ci, centroid in enumerate(centroids):
        sim = float(np.dot(vec, centroid))
        if sim > best_sim:
          best_sim = sim
          best_cluster = ci

      if best_sim >= self.config.similarity_threshold and best_cluster >= 0:
        # 归入已有簇，更新质心（增量平均 + 重新归一化）
        cluster_indices[best_cluster].append(idx)
        n = len(cluster_indices[best_cluster])
        new_centroid = centroids[best_cluster] * ((n - 1) / n) + vec / n
        norm = np.linalg.norm(new_centroid)
        centroids[best_cluster] = new_centroid / norm if norm > 0 else new_centroid
      else:
        # 新建簇
        cluster_indices.append([idx])
        centroids.append(vec.copy())

    # 转换为 CommentCluster
    clusters = []
    ungrouped = []
    for indices in cluster_indices:
      if len(indices) >= self.config.min_cluster_size:
        members = [comments[i] for i in indices]
        rep = max(members, key=lambda c: len(c.content))
        clusters.append(CommentCluster(
          representative=rep,
          members=tuple(members),
          merge_reason="semantic",
        ))
      else:
        for i in indices:
          ungrouped.append(comments[i])

    return clusters, ungrouped

  @staticmethod
  def _extract_repeating_unit(s: str) -> str:
    """
    提取最小循环节

    "哈哈哈哈" → "哈"
    "666666" → "6"
    "233233" → "233"
    "hello" → "hello"（无循环，返回自身）
    "" → ""
    """
    if not s:
      return s
    n = len(s)
    for period in range(1, n // 2 + 1):
      if n % period == 0 and s[:period] * (n // period) == s:
        return s[:period]
    return s
