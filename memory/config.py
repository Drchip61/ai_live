"""
记忆系统全局配置
所有可调常量汇总在此，方便调整和测试
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActiveConfig:
  """active 层配置"""
  capacity: int = 5  # FIFO 容量（条数）


@dataclass(frozen=True)
class TemporaryConfig:
  """temporary 层配置"""
  significance_threshold: float = 0.100  # 低于此值的记忆将被删除
  decay_coefficient: float = 0.9       # 未被取用时 significance 乘以此系数衰减


@dataclass(frozen=True)
class SummaryConfig:
  """summary 层配置"""
  interval_seconds: float = 60.0        # 汇总触发间隔（秒），默认 5 分钟
  significance_threshold: float = 0.050  # 低于此值的记忆将被删除
  decay_coefficient: float = 0.980       # 未被取用时 significance 乘以此系数衰减
  cleanup_interval_seconds: float = 600.0  # 清理触发间隔（秒），默认 10 分钟
  cleanup_ratio: float = 0.01           # 每次清理删除的比例（向下取整），默认 1%


@dataclass(frozen=True)
class RetrievalConfig:
  """跨层检索配置"""
  # 检索模式: "quota" = 每层定额, "weighted" = 加权合并
  mode: str = "quota"

  # quota 模式：每层取回数量
  quota_active: int = 0      # active 层无 RAG，全量直接注入
  quota_temporary: int = 3
  quota_summary: int = 2
  quota_static: int = 2

  # weighted 模式参数
  weight_temporary: float = 1.0
  weight_summary: float = 0.8
  weight_static: float = 1.2
  weighted_overfetch_multiplier: int = 3  # 多取回的倍数，再加权重排

  # 在 active 层格式化时引用主播回复原文
  include_response_in_active: bool = True
  # 在 temporary 层格式化时引用主播回复原文（预留，暂不启用）
  include_response_in_temporary: bool = False
  # 格式化时回复原文的最大显示长度
  response_display_max_length: int = 80


@dataclass(frozen=True)
class EmbeddingConfig:
  """嵌入模型配置"""
  model_name: str = "BAAI/bge-small-zh-v1.5"
  persist_directory: Optional[str] = "data/memory_store"


# 静态记忆类别定义：category -> 检索时添加的前缀
STATIC_CATEGORY_PREFIXES = {
  "identity": "【关于我自己的回忆】",
  "relationship": "【关于我认识的其他人的回忆】",
  "experience": "【让我联想起自己过去的回忆】",
  "personality": "【我对此的本能感觉与反应】",
  "world": "【我所知道的相关知识】",
}
STATIC_CATEGORY_DEFAULT_PREFIX = "【关于我自己的回忆】"


@dataclass(frozen=True)
class UserProfileConfig:
  """用户画像层配置"""
  enabled: bool = False
  persist_filename: str = "user_profile.json"


@dataclass(frozen=True)
class CharacterProfileConfig:
  """角色设定档层配置"""
  enabled: bool = False
  persist_filename: str = "character_profile.json"


@dataclass(frozen=True)
class MemoryConfig:
  """记忆系统总配置"""
  active: ActiveConfig = ActiveConfig()
  temporary: TemporaryConfig = TemporaryConfig()
  summary: SummaryConfig = SummaryConfig()
  retrieval: RetrievalConfig = RetrievalConfig()
  embedding: EmbeddingConfig = EmbeddingConfig()
  user_profile: UserProfileConfig = UserProfileConfig()
  character_profile: CharacterProfileConfig = CharacterProfileConfig()
