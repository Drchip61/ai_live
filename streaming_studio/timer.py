"""
管线阶段耗时统计

为主循环每一轮提供细粒度的阶段耗时记录，
支持打印摘要和通过 debug_state() 暴露给监控面板。

用法:
  timer.start_round()
  timer.mark("阶段A")
  # ... do stage A ...
  timer.mark("阶段B")
  # ... do stage B ...
  timings = timer.finish()
  print(timings.format_summary())
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

_BEIJING_TZ = timezone(timedelta(hours=8))


@dataclass
class StageTiming:
  """单个阶段的详细耗时记录"""
  name: str
  duration_ms: float
  started_at: datetime
  ended_at: datetime


@dataclass
class RoundTimings:
  """一轮管线各阶段耗时记录"""
  round_id: int
  stages: list[tuple[str, float]]
  total_ms: float
  timestamp: datetime
  skipped: bool = False
  started_at: Optional[datetime] = None
  ended_at: Optional[datetime] = None
  stage_details: list[StageTiming] = field(default_factory=list)

  def format_summary(self) -> str:
    """格式化为单行摘要"""
    parts = [f"[耗时] R{self.round_id}"]
    if self.skipped:
      parts[0] += "(跳过)"
    for name, ms in self.stages:
      if ms >= 1000:
        parts.append(f"{name}:{ms/1000:.1f}s")
      else:
        parts.append(f"{name}:{ms:.0f}ms")
    parts.append(f"合计:{self.total_ms/1000:.1f}s")
    return " | ".join(parts)


class PipelineTimer:
  """
  管线阶段计时器

  通过 start_round / mark / finish 三步记录每轮各阶段耗时。
  mark() 结束当前阶段并开始下一阶段，finish() 结束最后一个阶段并归档。
  """

  def __init__(self, history_maxlen: int = 100):
    self._round_id: int = 0
    self._last_finished_round_id: int = 0
    self._start: float = 0
    self._stage_start: float = 0
    self._round_started_at: Optional[datetime] = None
    self._stage_started_at: Optional[datetime] = None
    self._current_stage: str = ""
    self._stages: list[tuple[str, float]] = []
    self._stage_details: list[StageTiming] = []
    self._history: deque[RoundTimings] = deque(maxlen=history_maxlen)

  def start_round(self) -> None:
    """开始新一轮计时"""
    self._round_id += 1
    self._start = time.monotonic()
    self._stage_start = self._start
    now = datetime.now(_BEIJING_TZ)
    self._round_started_at = now
    self._stage_started_at = now
    self._current_stage = ""
    self._stages = []
    self._stage_details = []

  def mark(self, stage_name: str) -> None:
    """结束当前阶段并开始新阶段"""
    now = time.monotonic()
    now_wall = datetime.now(_BEIJING_TZ)
    if self._current_stage:
      duration_ms = (now - self._stage_start) * 1000
      self._stages.append((self._current_stage, duration_ms))
      self._stage_details.append(StageTiming(
        name=self._current_stage,
        duration_ms=duration_ms,
        started_at=self._stage_started_at or now_wall,
        ended_at=now_wall,
      ))
    self._stage_start = now
    self._stage_started_at = now_wall
    self._current_stage = stage_name

  def finish(self, skipped: bool = False) -> RoundTimings:
    """结束本轮计时，返回完整耗时记录"""
    if (
      self._last_finished_round_id == self._round_id
      and self._current_stage == ""
      and not self._stages
      and self.last is not None
    ):
      return self.last

    now = time.monotonic()
    now_wall = datetime.now(_BEIJING_TZ)
    if self._current_stage:
      duration_ms = (now - self._stage_start) * 1000
      self._stages.append((self._current_stage, duration_ms))
      self._stage_details.append(StageTiming(
        name=self._current_stage,
        duration_ms=duration_ms,
        started_at=self._stage_started_at or now_wall,
        ended_at=now_wall,
      ))
    total_ms = (now - self._start) * 1000
    timings = RoundTimings(
      round_id=self._round_id,
      stages=list(self._stages),
      total_ms=total_ms,
      timestamp=now_wall,
      skipped=skipped,
      started_at=self._round_started_at,
      ended_at=now_wall,
      stage_details=list(self._stage_details),
    )
    self._history.append(timings)
    self._last_finished_round_id = self._round_id
    self._current_stage = ""
    self._stages = []
    self._stage_details = []
    self._stage_started_at = None
    return timings

  @property
  def round_id(self) -> int:
    return self._round_id

  @property
  def history(self) -> list[RoundTimings]:
    return list(self._history)

  @property
  def last(self) -> Optional[RoundTimings]:
    return self._history[-1] if self._history else None

  def debug_state(self) -> dict:
    """调试状态快照"""
    last = self.last
    completed = [t for t in self._history if not t.skipped]
    if not last:
      return {"total_rounds": 0, "completed_rounds": 0}

    state: dict = {
      "total_rounds": self._round_id,
      "completed_rounds": len(completed),
      "skipped_rounds": self._round_id - len(completed),
    }

    if last:
      state["last_round"] = {
        "round_id": last.round_id,
        "total_ms": round(last.total_ms, 1),
        "stages": {name: round(ms, 1) for name, ms in last.stages},
        "skipped": last.skipped,
      }

    if completed:
      avg = sum(t.total_ms for t in completed) / len(completed)
      state["avg_total_ms"] = round(avg, 1)

      stage_totals: dict[str, list[float]] = {}
      for t in completed:
        for name, ms in t.stages:
          stage_totals.setdefault(name, []).append(ms)
      state["avg_stages_ms"] = {
        name: round(sum(vals) / len(vals), 1)
        for name, vals in stage_totals.items()
      }

    return state
