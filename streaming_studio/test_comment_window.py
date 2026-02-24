"""
回归测试：防止“回复期间弹幕被吃掉”。

核心预期：
- 以“开始回复时刻”作为新旧分界；
- 在回复期间到达的弹幕，下一轮依然会被识别为新弹幕。
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from streaming_studio import StreamingStudio
from streaming_studio.models import Comment
from langchain_wrapper import ModelType


def _make_comment(content: str, ts: datetime, priority: bool = False) -> Comment:
  return Comment(
    user_id="u",
    nickname="n",
    content=content,
    timestamp=ts,
    priority=priority,
  )


def test_comments_during_reply_should_not_be_dropped() -> None:
  """
  模拟时间线：
  - t1: 上一轮已回复完成（并不影响分界）
  - t2: 本轮开始回复（分界点）
  - t3: 回复期间来了新弹幕

  预期：
  - t3 弹幕在下一轮 _collect_comments() 中属于 new_comments。
  """
  studio = StreamingStudio(
    persona="kuro",
    model_type=ModelType.OPENAI,
    enable_reply_decider=False,
  )

  base = datetime.now()
  t1 = base - timedelta(seconds=5)
  t2 = base - timedelta(seconds=2)  # 上轮开始回复时间（分界点）
  t3 = base - timedelta(seconds=1)  # 回复期间到达的新弹幕

  # 旧实现常见误用是按 last_reply_time 分割，这里故意设置为更晚时间
  # 来验证“以 last_collect_time 为准”。
  studio._last_reply_time = base
  studio._last_collect_time = t2

  studio._comment_buffer.clear()
  studio._comment_buffer.append(_make_comment("old-before-collect", t1))
  studio._comment_buffer.append(_make_comment("new-during-reply", t3))

  old_comments, new_comments = studio._collect_comments()
  new_contents = [c.content for c in new_comments]
  old_contents = [c.content for c in old_comments]

  assert "new-during-reply" in new_contents
  assert "new-during-reply" not in old_contents

