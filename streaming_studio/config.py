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
  recent_comments_limit: int = 30
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

  engaging_question_probability: float = 0.3
  """引导式反问概率：每轮回复有此概率追加引导反问 prompt，增强观众互动"""

  speech_queue_enabled: bool = True
  """是否启用 SpeechQueue 双循环架构（False 时走旧 _main_loop 串行路径）"""

  controller_url: str = ""
  """LLM Controller 的 OpenAI 兼容接口地址（如 http://localhost:2001/v1）。为空则不启用 Controller。"""

  controller_model: str = "qwen3.5-9b"
  """Controller 使用的模型名称"""


@dataclass(frozen=True)
class SpeechQueueConfig:
  """语音调度队列配置"""

  max_size: int = 4
  """队列最大容量（超出时驱逐优先级最低的条目）"""

  paid_event_ttl: float = 120.0
  """付费事件（SC / 上舰 / >=5元礼物）的生存时间（秒）"""

  danmaku_ttl: float = 25.0
  """普通弹幕回复的生存时间（秒）"""

  game_ttl: float = 60.0
  """游戏解说（外部推送）的生存时间（秒）"""

  event_low_ttl: float = 20.0
  """低优先级事件（小礼物 / 入场问候）的生存时间（秒）"""

  video_ttl: float = 10.0
  """视频解说 / 独白的生存时间（秒）"""


@dataclass(frozen=True)
class CommentClustererConfig:
  """弹幕聚类器配置"""

  similarity_threshold: float = 0.75
  """语义相似度阈值（cosine similarity），高于此值合并"""

  min_cluster_size: int = 2
  """最小成簇数量，低于此值不合并"""

  max_pattern_unit_length: int = 4
  """规则阶段：循环节最大长度（超过此长度不视为循环模式）"""


@dataclass(frozen=True)
class SceneMemoryConfig:
  """
  场景记忆缓存配置

  用小模型 VLM 异步描述视频帧，维护内存滚动缓冲，
  为主模型提供时序上下文（"之前发生了什么"），
  使 AI 对画面的反应具有叙事连续性和情感投入。
  """

  enabled: bool = True
  """是否启用场景记忆"""

  buffer_size: int = 10
  """滚动缓冲区容量（保留最近 N 条场景描述）"""

  min_describe_interval: float = 10.0
  """两次场景描述之间的最小间隔（秒，视频时间）"""

  change_threshold: float = 0.85
  """直方图相关性阈值：低于此值视为画面变化显著。1.0=完全相同，0.0=完全不同"""

  max_prompt_items: int = 5
  """注入 prompt 的最近场景条数"""
