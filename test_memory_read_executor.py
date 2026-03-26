"""
前台记忆读取 executor 回归测试
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from langchain_wrapper.retriever import RetrieverResolver
from memory.manager import MemoryManager
from streaming_studio.models import Comment


def test_retriever_prefers_memory_async_read_methods():
  """Retriever 应优先走 memory_manager 的异步读接口，而不是裸 to_thread。"""

  async def scenario():
    calls: list[object] = []

    class FakeMemoryManager:
      async def retrieve_active_only_async(self):
        calls.append("active_async")
        return "【近期记忆】\n- 上次聊到周杰伦新歌", "", ""

      def retrieve_active_only(self):
        raise AssertionError("不应回退到同步 retrieve_active_only")

      async def compile_structured_context_async(
        self,
        query="",
        viewer_ids=None,
        include_persona=True,
        include_corpus=False,
        include_external_knowledge=False,
        recall_profile="deep_recall",
      ):
        calls.append(("structured_async", query, tuple(viewer_ids or ()), recall_profile))
        return "【结构化记忆】\n- 你答应过补聊副歌那段"

      def compile_structured_context(self, *args, **kwargs):
        raise AssertionError("不应回退到同步 compile_structured_context")

    resolver = RetrieverResolver(memory_manager=FakeMemoryManager())
    plan = SimpleNamespace(
      route_kind="chat",
      memory_strategy="normal",
      viewer_focus_ids=(),
      persona_sections=(),
      knowledge_topics=(),
      corpus_style="",
      corpus_scene="",
      extra_instructions=(),
    )
    bundle = await resolver.resolve(
      plan,
      old_comments=[],
      new_comments=[
        Comment(
          user_id="u1",
          nickname="观众A",
          content="你还记得上次那首周杰伦的新歌吗",
        )
      ],
      viewer_ids=["u1"],
    )

    assert "active_async" in calls
    assert any(
      isinstance(call, tuple) and call[0] == "structured_async"
      for call in calls
    )
    assert bundle.effective_memory_strategy == "deep_recall"
    assert "active_memory" in bundle.debug_view()["untrusted_sources"]
    assert "structured_memory" in bundle.debug_view()["untrusted_sources"]

  asyncio.run(scenario())
  print("  [PASS] Retriever 已优先使用 memory_manager 异步读接口")


def test_memory_manager_async_read_helpers_use_read_backlog_channel():
  """MemoryManager 的异步读包装应统一走 _read_backlog 统计。"""

  async def scenario():
    manager = MemoryManager.__new__(MemoryManager)
    manager._read_executor = object()
    manager._read_backlog = 0
    calls: list[tuple[object, str, tuple]] = []

    async def fake_run_executor_job(executor, backlog_attr, callback, *args):
      calls.append((executor, backlog_attr, args))
      return callback(*args)

    manager._run_executor_job = fake_run_executor_job
    manager.retrieve_active_only = lambda: ("【近期记忆】\n- test", "", "")
    manager.compile_structured_context = lambda *args: "【结构化记忆】\n- test"

    active = await MemoryManager.retrieve_active_only_async(manager)
    structured = await MemoryManager.compile_structured_context_async(
      manager,
      "周杰伦",
      ["u1"],
      False,
      False,
      False,
      "deep_recall",
    )

    assert active[0].startswith("【近期记忆】")
    assert structured.startswith("【结构化记忆】")
    assert len(calls) == 2
    assert calls[0][1] == "_read_backlog"
    assert calls[1][1] == "_read_backlog"

  asyncio.run(scenario())
  print("  [PASS] MemoryManager 异步读包装会统计 memory_read_queue")


def test_memory_manager_structured_trace_uses_read_backlog_channel():
  """带 trace 的 structured 读取也应复用 _read_backlog，并产出 queue/exec timing。"""

  async def scenario():
    manager = MemoryManager.__new__(MemoryManager)
    manager._read_executor = object()
    manager._read_backlog = 0
    calls: list[tuple[object, str, tuple]] = []

    async def fake_run_executor_job_with_timing(executor, backlog_attr, callback, *args):
      calls.append((executor, backlog_attr, args))
      return callback(*args), {
        "queue_wait_ms": 12.3,
        "exec_ms": 45.6,
      }

    manager._run_executor_job_with_timing = fake_run_executor_job_with_timing
    manager.compile_structured_context_with_trace = lambda *args: (
      "【结构化记忆】\n- test",
      {"semantic_search_count": 2, "query_embed_count": 1},
    )

    structured, trace = await MemoryManager.compile_structured_context_with_trace_async(
      manager,
      "周杰伦",
      ["u1"],
      False,
      False,
      False,
      "deep_recall",
    )

    assert structured.startswith("【结构化记忆】")
    assert trace["semantic_search_count"] == 2
    assert trace["read_queue_wait_ms"] == 12.3
    assert trace["read_exec_ms"] == 45.6
    assert len(calls) == 1
    assert calls[0][1] == "_read_backlog"

  asyncio.run(scenario())
  print("  [PASS] MemoryManager trace 读取会统计 queue/exec timing")


def main():
  tests = [
    test_retriever_prefers_memory_async_read_methods,
    test_memory_manager_async_read_helpers_use_read_backlog_channel,
    test_memory_manager_structured_trace_uses_read_backlog_channel,
  ]

  failed = 0
  for test_fn in tests:
    try:
      test_fn()
    except AssertionError as e:
      failed += 1
      print(f"  [FAIL] {test_fn.__name__}: {e}")
    except Exception as e:
      failed += 1
      print(f"  [ERROR] {test_fn.__name__}: {type(e).__name__}: {e}")

  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
