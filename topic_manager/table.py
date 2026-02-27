"""
话题表
纯内存存储，提供话题的 CRUD 和查询操作
"""

import logging
from dataclasses import replace
from datetime import datetime
from typing import Optional

from .config import TopicManagerConfig
from .models import Topic

logger = logging.getLogger(__name__)


class TopicTable:
  """
  内存话题表

  使用 dict[str, Topic] 存储，所有更新操作创建新 Topic 实例（不可变）。
  """

  def __init__(self, config: TopicManagerConfig):
    self._config = config
    self._topics: dict[str, Topic] = {}

  def add(self, topic: Topic) -> None:
    """
    添加话题

    如果超过 max_topics，删除 significance 最低的话题。

    Args:
      topic: 话题实例
    """
    self._topics[topic.topic_id] = topic

    # 超出上限时删除最低 significance 的话题
    while len(self._topics) > self._config.max_topics:
      lowest = min(self._topics.values(), key=lambda t: t.significance)
      del self._topics[lowest.topic_id]
      logger.debug("话题表已满，删除最低话题: %s", lowest.topic_id)

  def remove(self, topic_id: str) -> Optional[Topic]:
    """
    删除话题

    Args:
      topic_id: 话题 ID

    Returns:
      被删除的话题，不存在返回 None
    """
    return self._topics.pop(topic_id, None)

  def get(self, topic_id: str) -> Optional[Topic]:
    """
    获取话题

    Args:
      topic_id: 话题 ID

    Returns:
      话题实例，不存在返回 None
    """
    return self._topics.get(topic_id)

  def update(self, topic_id: str, **kwargs) -> Optional[Topic]:
    """
    更新话题（创建新实例替换）

    自动截断 comment_ids 和 user_ids，自动更新 updated_at。

    Args:
      topic_id: 话题 ID
      **kwargs: 要更新的字段

    Returns:
      更新后的话题，不存在返回 None
    """
    old = self._topics.get(topic_id)
    if old is None:
      return None

    # 自动更新时间戳
    if "updated_at" not in kwargs:
      kwargs["updated_at"] = datetime.now()

    new_topic = replace(old, **kwargs)

    # 截断列表
    if len(new_topic.comment_ids) > self._config.max_comment_ids_per_topic:
      new_topic = replace(
        new_topic,
        comment_ids=new_topic.comment_ids[-self._config.max_comment_ids_per_topic:],
      )
    if len(new_topic.user_ids) > self._config.max_user_ids_per_topic:
      new_topic = replace(
        new_topic,
        user_ids=new_topic.user_ids[-self._config.max_user_ids_per_topic:],
      )

    self._topics[topic_id] = new_topic
    return new_topic

  def add_comment_to_topic(
    self,
    topic_id: str,
    comment_id: str,
    user_id: str,
  ) -> Optional[Topic]:
    """
    将弹幕和用户关联到话题

    用户 ID 重复出现时刷新位置（移到最后）。

    Args:
      topic_id: 话题 ID
      comment_id: 弹幕 ID
      user_id: 用户 ID

    Returns:
      更新后的话题，不存在返回 None
    """
    old = self._topics.get(topic_id)
    if old is None:
      return None

    # 追加弹幕 ID（不去重，保留时间顺序）
    new_comment_ids = old.comment_ids + (comment_id,)

    # 用户 ID 去重后追加到末尾（刷新位置）
    existing_users = tuple(uid for uid in old.user_ids if uid != user_id)
    new_user_ids = existing_users + (user_id,)

    return self.update(
      topic_id,
      comment_ids=new_comment_ids,
      user_ids=new_user_ids,
      last_discussed_at=datetime.now(),
    )

  def get_top_k(self, k: int) -> list[Topic]:
    """
    按 significance 排序取前 k 个

    Args:
      k: 数量

    Returns:
      按 significance 降序排列的话题列表
    """
    sorted_topics = sorted(
      self._topics.values(),
      key=lambda t: t.significance,
      reverse=True,
    )
    return sorted_topics[:k]

  def get_by_comment(self, comment_id: str) -> Optional[Topic]:
    """
    查找包含该弹幕的话题

    Args:
      comment_id: 弹幕 ID

    Returns:
      话题实例，未找到返回 None
    """
    for topic in self._topics.values():
      if comment_id in topic.comment_ids:
        return topic
    return None

  def get_all(self) -> list[Topic]:
    """获取所有话题"""
    return list(self._topics.values())

  def count(self) -> int:
    """获取话题数量"""
    return len(self._topics)

  def decay_all(self, coefficient: float) -> None:
    """
    全部话题 significance 衰减

    Args:
      coefficient: 衰减系数
    """
    for topic_id, topic in list(self._topics.items()):
      new_sig = round(topic.significance * coefficient, 3)
      self._topics[topic_id] = replace(topic, significance=new_sig)

  def idle_decay(
    self,
    threshold_seconds: float,
    decay_rate: float,
  ) -> None:
    """
    对空闲话题施加额外 significance 衰减

    超过 threshold_seconds 未收到弹幕的话题，
    每超过 60 秒额外扣减 decay_rate。

    Args:
      threshold_seconds: 空闲衰减起始阈值（秒）
      decay_rate: 每超过 60 秒的额外衰减量
    """
    now = datetime.now()
    for topic_id, topic in list(self._topics.items()):
      idle_seconds = (now - topic.last_discussed_at).total_seconds()
      if idle_seconds <= threshold_seconds:
        continue
      excess = idle_seconds - threshold_seconds
      penalty = decay_rate * (excess / 60.0)
      new_sig = round(max(0.0, topic.significance - penalty), 3)
      if new_sig != topic.significance:
        self._topics[topic_id] = replace(topic, significance=new_sig)

  def boost(self, topic_id: str, boost_factor: float) -> Optional[Topic]:
    """
    提升话题 significance

    算法：new = current + (1 - current) * boost_factor

    Args:
      topic_id: 话题 ID
      boost_factor: 提升系数

    Returns:
      更新后的话题，不存在返回 None
    """
    old = self._topics.get(topic_id)
    if old is None:
      return None

    new_sig = round(old.significance + (1.0 - old.significance) * boost_factor, 3)
    return self.update(topic_id, significance=new_sig)

  def cleanup(self, threshold: float) -> int:
    """
    删除低于阈值的话题

    Args:
      threshold: significance 阈值

    Returns:
      删除的话题数量
    """
    to_remove = [
      tid for tid, t in self._topics.items()
      if t.significance < threshold
    ]
    for tid in to_remove:
      del self._topics[tid]
    if to_remove:
      logger.debug("清理了 %d 个低 significance 话题", len(to_remove))
    return len(to_remove)
