"""
StreamingStudio 配置
管理直播间行为相关的细节参数
"""

from dataclasses import dataclass, field


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

  tts_skip_timer_threshold: int = 3
  """TTS 播放期间积累弹幕数 >= 此值时，完播后跳过定时器直接进入回复流程"""

  proactive_fallback_silence_multiplier: float = 1.5
  """非对话模式下话题耗尽时的兜底主动发言沉默倍数（基于 proactive_silence_threshold）"""

  comment_priority_mode: bool = True
  """弹幕优先模式：有新弹幕时跳过画面传递，专注弹幕互动；无弹幕时走 VLM 画面理解"""

  engaging_question_probability: float = 0.3
  """引导式反问概率：每轮回复有此概率追加引导反问 prompt，增强观众互动"""

  speech_queue_enabled: bool = True
  """是否启用 SpeechQueue 双循环架构（False 时走旧 _main_loop 串行路径）"""


@dataclass(frozen=True)
class SpeechQueueConfig:
  """语音调度队列配置"""

  max_size: int = 4
  """队列最大容量（超出时驱逐优先级最低的条目）"""

  paid_event_ttl: float = 120.0
  """付费事件（SC / 上舰 / >=5元礼物）的生存时间（秒）"""

  danmaku_ttl: float = 30.0
  """普通弹幕回复的生存时间（秒）"""

  session_cont_ttl: float = 20.0
  """CommentSession 续接句的生存时间（秒）"""

  event_low_ttl: float = 20.0
  """低优先级事件（小礼物 / 入场问候）的生存时间（秒）"""

  video_ttl: float = 10.0
  """视频解说 / 独白的生存时间（秒）"""

  paid_gift_threshold: float = 5.0
  """礼物价格 >= 此值视为付费事件（priority 0），低于走 priority 2"""

  monologue_interval: float = 20.0
  """独白/视频解说的最小 VLM 调用间隔（秒）"""


@dataclass(frozen=True)
class ReplyDeciderConfig:
  """回复决策器配置"""

  min_quality_length: int = 3
  """低于此长度的弹幕视为低质量（纯反应词）"""

  must_reply_comment_count: int = 5
  """新弹幕数量 >= 此值时规则直接放行"""

  skip_patterns: tuple[str, ...] = (
    "哈哈", "哈哈哈", "hhhh", "hhh", "233", "666", "nb",
    "草", "笑死", "www", "awsl", "yyds", "dd", "弹幕",
    "1", "11", "111", "??", "？？",
  )
  """这些模式如果覆盖所有弹幕则建议跳过"""

  proactive_silence_threshold: float = 10.0
  """沉默超过此秒数后主动发言（优先基于画面变化判断）"""

  llm_judge_urgency_threshold: float = 4.0
  """LLM 精判返回 urgency 低于此值时跳过回复"""

  sparse_chat_threshold: float = 5.0
  """弹幕速率（条/分钟）低于此值时进入稀疏模式，放宽过滤标准"""

  very_sparse_threshold: float = 1.0
  """弹幕速率（条/分钟）低于此值时极稀疏模式，规则层直接放行不走 LLM"""


@dataclass(frozen=True)
class EventResponderConfig:
  """事件模板回复配置（GUARD_BUY / GIFT / ENTRY 统一管理）"""

  enabled: bool = True
  """是否启用模板回复"""

  # 各事件冷却间隔
  guard_cooldown: float = 0.0
  """上舰感谢冷却（秒），0 表示立即回复"""

  gift_cooldown: float = 5.0
  """礼物感谢最小间隔（秒）"""

  entry_cooldown: float = 20.0
  """普通入场问候最小间隔（秒）"""

  # 过期清理 TTL
  gift_ttl: float = 120.0
  """礼物事件过期时间（秒），超时自动丢弃"""

  entry_ttl: float = 60.0
  """普通入场事件过期时间（秒），超时自动丢弃"""

  # 入场批量参数
  entry_max_names: int = 3
  """单次入场问候最多念出的昵称数，超出用"等N位"省略"""

  entry_crowd_threshold: int = 6
  """超过此人数直接用 crowd 模板（不念名字）"""

  # 礼物价格分档阈值（元）：free / small / medium / large
  gift_tier_thresholds: tuple[float, ...] = (0.0, 0.1, 10.0, 100.0)
  """礼物分档边界：[0, 0.1) = free, [0.1, 10) = small, [10, 100) = medium, [100, +∞) = large"""

  # 前缀模式（便宜礼物 + 普通入场附带在 LLM 回复前播报）
  prefix_gift_threshold: float = 10.0
  """礼物价格低于此值走前缀模式，高于走独立回复（元）"""

  prefix_cooldown: float = 30.0
  """前缀模式冷却时间（秒），约每 2-3 条回复最多带一次问候"""

  # 多句感谢（重要事件拼接多条模板）
  guard_min_sentences: int = 3
  """上舰感谢最少句数（拼接多条模板）"""

  large_gift_min_sentences: int = 3
  """大额礼物（>=100元）感谢最少句数"""

  medium_gift_min_sentences: int = 2
  """中等礼物（10-100元）感谢最少句数"""

  # 模板去重
  recent_dedup_count: int = 8
  """记住最近 N 次使用的模板索引，避免连续重复"""


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
class SessionConfig:
  """
  聚焦会话配置

  控制 FocusSession 的触发条件、持续时长和行为参数。
  两种会话类型：CommentSession（弹幕聚焦）和 VideoSession（看视频聚焦）。
  """

  enabled: bool = True
  """是否启用聚焦会话系统"""

  # CommentSession 参数
  comment_session_max_rounds: int = 4
  """弹幕聚焦会话最大轮数"""

  # VideoSession 参数
  video_session_max_rounds: int = 3
  """看视频聚焦会话最大轮数"""

  video_session_silence_threshold: float = 30.0
  """看视频会话触发所需的最低沉默时长（秒）"""

  video_session_min_scene_changes: int = 2
  """触发看视频会话前需要的最少场景变化次数"""

  # 高质量弹幕判定
  min_quality_length: int = 8
  """观点/讨论类弹幕的最低字数"""

  question_min_length: int = 6
  """提问类弹幕去掉问号后的最低字数"""

  long_content_length: int = 3
  """非噪声弹幕触发 CommentSession 的最低字数"""

  # Session 生命周期
  relevance_timeout: float = 60.0
  """session 无相关弹幕超过此秒数后自动关闭"""

  stale_rounds_to_end: int = 4
  """连续无相关弹幕的轮数达到此值时关闭 session"""

  # 节奏调整
  session_min_interval: float = 2.0
  """session 活跃时的回复最小间隔（秒），比默认更快"""

  session_max_interval: float = 6.0
  """session 活跃时的回复最大间隔（秒），比默认更快"""

  # prompt 句数覆盖
  comment_session_sentences: int = 2
  """弹幕聚焦会话时允许的句数（比默认多说一些）"""

  video_session_sentences: int = 2
  """看视频会话时允许的句数"""


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
