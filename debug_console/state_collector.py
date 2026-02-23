"""
状态收集器
从各模块聚合 debug_state，供监控面板使用
"""

from streaming_studio import StreamingStudio


class StateCollector:
  """
  聚合各模块的 debug_state() 输出为统一快照

  只读操作，不修改任何模块状态。
  """

  def __init__(self, studio: StreamingStudio):
    self._studio = studio

  def snapshot(self) -> dict:
    """
    收集一次完整的状态快照

    Returns:
      {
        "studio": {...},
        "llm": {...},
        "memory": {...} | None,
      }
    """
    try:
      studio_state = self._studio.debug_state()
    except Exception:
      studio_state = {}

    try:
      llm_state = self._studio.llm_wrapper.debug_state()
    except Exception:
      llm_state = {}

    try:
      memory_state = self._studio.llm_wrapper.memory_debug_state()
    except Exception:
      memory_state = None

    try:
      topic_state = self._studio.topic_debug_state()
    except Exception:
      topic_state = None

    return {
      "studio": studio_state,
      "llm": llm_state,
      "memory": memory_state,
      "topics": topic_state,
    }
