"""
StreamingStudio 配置
管理直播间行为相关的细节参数
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StudioConfig:
  """
  StreamingStudio 行为配置

  包含回复节奏、缓冲区大小等细节参数，
  与核心配置（persona、model_type）区分开。
  """

  # 双轨定时器参数
  min_interval: float = 3.0
  """回复最小间隔（秒）"""

  max_interval: float = 10.0
  """回复最大间隔（秒）"""

  # 弹幕处理参数
  recent_comments_limit: int = 20
  """每次回复时收集的最近弹幕数上限"""

  buffer_maxlen: int = 200
  """弹幕缓冲区最大容量（环形队列）"""

  comment_wait_reduction: float = 0.1
  """每条新弹幕减少的等待时间（秒）"""

  new_comment_context_ratio: float = 1.0
  """实际送入模型的弹幕上限 = min(recent_comments_limit, 新弹幕数 * 此系数)"""

  # 互动目标选择参数
  interaction_target_mu: float = 2.5
  """选中互动目标数量的高斯均值"""

  interaction_target_sigma: float = 1.0
  """选中互动目标数量的高斯标准差"""

  interaction_base_weight: float = 0.3
  """无话题归属弹幕的基础权重"""

  interaction_stale_weight: float = 0.1
  """过期话题弹幕的权重"""
