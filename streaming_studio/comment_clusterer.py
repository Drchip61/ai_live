"""
弹幕聚类器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import CommentClustererConfig
from .models import Comment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommentCluster:
  """单个弹幕簇"""

  representative: Comment
  members: tuple[Comment, ...] = field(default_factory=tuple)
  merge_reason: str = "pattern"

  @property
  def count(self) -> int:
    return len(self.members)


@dataclass(frozen=True)
class ClusterResult:
  """弹幕聚类结果"""

  clusters: tuple[CommentCluster, ...] = field(default_factory=tuple)
  singles: tuple[Comment, ...] = field(default_factory=tuple)

  def representatives(self) -> list[Comment]:
    """去重后的代表弹幕列表（单条 + 每簇代表），按时间排序"""
    result = list(self.singles)
    for cluster in self.clusters:
      result.append(cluster.representative)
    return sorted(result, key=lambda c: c.timestamp)

  def cluster_for(self, comment_id: str) -> Optional[CommentCluster]:
    """根据弹幕 ID 查找所属簇，不在任何簇中返回 None"""
    for cluster in self.clusters:
      if any(member.id == comment_id for member in cluster.members):
        return cluster
    return None


class CommentClusterer:
  """
  两阶段弹幕聚类器

  Phase 1（规则快筛）：
    提取每条弹幕的最小循环节，循环节相同的弹幕直接归组。

  Phase 2（语义聚类）：
    对规则阶段未归组的弹幕做 embedding + cosine similarity。
  """

  def __init__(
    self,
    config: Optional[CommentClustererConfig] = None,
    embeddings=None,
  ):
    self.config = config or CommentClustererConfig()
    self._embeddings = embeddings

  def cluster(self, comments: list[Comment]) -> ClusterResult:
    """完整两阶段聚类"""
    if not comments:
      return ClusterResult()

    priority = [c for c in comments if c.priority]
    normal = [c for c in comments if not c.priority]
    if not normal:
      return ClusterResult(singles=tuple(priority))

    rule_clusters, remaining = self._rule_cluster(normal)
    semantic_clusters, still_remaining = self._semantic_cluster(remaining)

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
    """Phase 1: 按循环节归组"""
    groups: dict[str, list[Comment]] = {}
    for comment in comments:
      content = comment.content.strip().lower()
      unit = self._extract_repeating_unit(content)
      key = content if len(unit) > self.config.max_pattern_unit_length else unit
      groups.setdefault(key, []).append(comment)

    clusters: list[CommentCluster] = []
    ungrouped: list[Comment] = []
    for members in groups.values():
      if len(members) >= self.config.min_cluster_size:
        representative = max(members, key=lambda c: len(c.content))
        clusters.append(CommentCluster(
          representative=representative,
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
    """Phase 2: 语义聚类（增量阈值法）"""
    if not comments or self._embeddings is None:
      return [], list(comments) if comments else []
    if len(comments) < self.config.min_cluster_size:
      return [], list(comments)

    try:
      texts = [c.content for c in comments]
      vectors = self._embeddings.embed_documents(texts)
      matrix = np.array(vectors, dtype=np.float32)
    except Exception as e:
      logger.warning("语义聚类 embedding 失败: %s", e)
      return [], list(comments)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    cluster_indices: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for idx in range(len(comments)):
      vec = matrix[idx]
      best_sim = -1.0
      best_cluster = -1

      for cluster_idx, centroid in enumerate(centroids):
        sim = float(np.dot(vec, centroid))
        if sim > best_sim:
          best_sim = sim
          best_cluster = cluster_idx

      if best_sim >= self.config.similarity_threshold and best_cluster >= 0:
        cluster_indices[best_cluster].append(idx)
        size = len(cluster_indices[best_cluster])
        new_centroid = centroids[best_cluster] * ((size - 1) / size) + vec / size
        norm = np.linalg.norm(new_centroid)
        centroids[best_cluster] = new_centroid / norm if norm > 0 else new_centroid
      else:
        cluster_indices.append([idx])
        centroids.append(vec.copy())

    clusters: list[CommentCluster] = []
    ungrouped: list[Comment] = []
    for indices in cluster_indices:
      if len(indices) >= self.config.min_cluster_size:
        members = [comments[i] for i in indices]
        representative = max(members, key=lambda c: len(c.content))
        clusters.append(CommentCluster(
          representative=representative,
          members=tuple(members),
          merge_reason="semantic",
        ))
      else:
        for idx in indices:
          ungrouped.append(comments[idx])

    return clusters, ungrouped

  @staticmethod
  def _extract_repeating_unit(text: str) -> str:
    """提取最小循环节"""
    if not text:
      return text
    length = len(text)
    for period in range(1, length // 2 + 1):
      if length % period == 0 and text[:period] * (length // period) == text:
        return text[:period]
    return text
