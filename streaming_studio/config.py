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

  tts_skip_timer_threshold: int = 3
  """TTS 播放期间积累弹幕数 >= 此值时，完播后跳过定时器直接进入回复流程"""

  proactive_fallback_silence_multiplier: float = 1.5
  """非对话模式下话题耗尽时的兜底主动发言沉默倍数（基于 proactive_silence_threshold）"""

  comment_priority_mode: bool = True
  """弹幕优先模式：有新弹幕时跳过画面传递，专注弹幕互动；无弹幕时走 VLM 画面理解"""

  engaging_question_probability: float = 0.3
  """引导式反问概率：每轮回复有此概率追加引导反问 prompt，增强观众互动"""


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
class CommentClustererConfig:
  """弹幕聚类器配置"""

  similarity_threshold: float = 0.75
  """语义相似度阈值（cosine similarity），高于此值合并"""

  min_cluster_size: int = 2
  """最小成簇数量，低于此值不合并"""

  max_pattern_unit_length: int = 4
  """规则阶段：循环节最大长度（超过此长度不视为循环模式）"""
