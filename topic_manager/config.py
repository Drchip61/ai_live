"""
话题管理器配置
所有可调参数汇总在此
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicManagerConfig:
  """话题管理器配置"""

  # 弹幕处理模式: "single" 每条触发 / "batch" 批量触发
  comment_mode: str = "batch"

  # 批量模式参数
  batch_wait_seconds: float = 10.0
  """最大等待时间（秒），到时间且至少有 1 条弹幕就触发"""

  batch_size_threshold: int = 5
  """达到此数量立即触发分类"""

  # 话题表参数
  max_topics: int = 10
  """最大活跃话题数"""

  initial_significance: float = 0.5
  """新话题初始重要性"""

  significance_decay: float = 0.9
  """每次回复后的衰减系数"""

  significance_boost: float = 0.3
  """弹幕提及话题时的 boost 增量（与 1 的差值乘此系数）"""

  significance_threshold: float = 0.1
  """低于此值删除话题"""

  # 输出参数
  top_k_topics: int = 3
  """prompt 中列出的最大话题数"""

  recent_comments_per_topic: int = 3
  """每个话题展示的最近弹幕数"""

  recent_users_per_topic: int = 3
  """每个话题展示的最近用户数"""

  # 列表截断
  max_comment_ids_per_topic: int = 20
  """每个话题保留的最大弹幕 ID 数"""

  max_user_ids_per_topic: int = 10
  """每个话题保留的最大用户 ID 数"""

  # 动态等待时间
  enable_dynamic_timing: bool = True
  """是否启用动态等待时间"""

  # 按需分析阈值
  min_comments_for_analysis: int = 2
  """新弹幕少于此数跳过内容分析（只做衰减）"""

  min_analysis_interval_seconds: float = 30.0
  """距上次分析不足此时间跳过模型调用"""

  # 时间感知：空闲话题额外衰减
  idle_decay_threshold_seconds: float = 60.0
  """话题空闲超过此秒数后开始额外衰减 significance"""

  idle_decay_rate: float = 0.02
  """每超过阈值 60 秒额外扣减此值（线性衰减）"""

  # 主动话题推进
  sparse_comment_threshold: int = 2
  """新弹幕数 <= 此值视为"稀疏"，触发话题跟进建议"""

  proactive_topic_min_significance: float = 0.25
  """主动推进话题的最低 significance 要求"""

  proactive_topic_min_idle_seconds: float = 30.0
  """话题至少空闲此秒数才值得主动推进（避免推荐刚聊过的话题）"""

  # 网络搜索（可选，默认关）
  enable_web_search: bool = False
  """是否启用网络搜索（Tavily）"""
