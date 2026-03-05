"""
舰长/提督/总督会员名册
持久化到 JSON 文件，自动过期清理。

用法:
  roster = GuardRoster("data/guard_roster.json")
  roster.add_or_extend("uid_123", "小明", guard_level=1, num_months=1)
  print(roster.is_member("uid_123"))        # True
  print(roster.get_level_name("uid_123"))   # "舰长"
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BJT = timezone(timedelta(hours=8))
_DAYS_PER_MONTH = 30
_LEVEL_NAMES = {1: "舰长", 2: "提督", 3: "总督"}


@dataclass
class GuardMember:
  nickname: str
  uid: str
  guard_level: int
  expiry_time: datetime
  first_joined: datetime

  def is_expired(self, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now(_BJT)
    return now >= self.expiry_time

  @property
  def level_name(self) -> str:
    return _LEVEL_NAMES.get(self.guard_level, "舰长")


class GuardRoster:
  """
  会员名册管理器

  以 uid 为主键，nickname 为展示字段。
  每次公开读写操作前自动清理过期会员。
  """

  def __init__(self, path: str = "data/guard_roster.json"):
    self._path = Path(path)
    self._members: dict[str, GuardMember] = {}
    self._load()

  # ── 公开接口 ──

  def add_or_extend(
    self,
    uid: str,
    nickname: str,
    guard_level: int,
    num_months: int = 1,
  ) -> GuardMember:
    """
    添加新会员或续期已有会员

    - 新会员: 从当前北京时间起算 num_months * 30 天
    - 已有会员: 在当前 expiry 基础上累加天数
    - 新等级高于旧等级时升级
    """
    now = datetime.now(_BJT)
    extend_days = num_months * _DAYS_PER_MONTH

    existing = self._members.get(uid)
    if existing and not existing.is_expired(now):
      new_expiry = existing.expiry_time + timedelta(days=extend_days)
      new_level = max(existing.guard_level, guard_level)
      member = GuardMember(
        nickname=nickname,
        uid=uid,
        guard_level=new_level,
        expiry_time=new_expiry,
        first_joined=existing.first_joined,
      )
      logger.info("会员续期: %s (%s) → %s, 等级=%s",
                  nickname, uid, new_expiry.isoformat(), _LEVEL_NAMES.get(new_level))
    else:
      member = GuardMember(
        nickname=nickname,
        uid=uid,
        guard_level=guard_level,
        expiry_time=now + timedelta(days=extend_days),
        first_joined=now,
      )
      logger.info("新会员: %s (%s), 等级=%s, 到期=%s",
                  nickname, uid, _LEVEL_NAMES.get(guard_level),
                  member.expiry_time.isoformat())

    self._members[uid] = member
    self._save()
    return member

  def get_member(self, uid: str) -> Optional[GuardMember]:
    self._cleanup()
    return self._members.get(uid)

  def is_member(self, uid: str) -> bool:
    return self.get_member(uid) is not None

  def get_level_name(self, uid: str) -> str:
    """返回会员等级中文名，非会员返回空字符串"""
    member = self.get_member(uid)
    return member.level_name if member else ""

  def get_all_active(self) -> list[GuardMember]:
    self._cleanup()
    return list(self._members.values())

  @property
  def member_count(self) -> int:
    self._cleanup()
    return len(self._members)

  # ── 持久化 ──

  def _save(self) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for uid, m in self._members.items():
      data[uid] = {
        "nickname": m.nickname,
        "guard_level": m.guard_level,
        "expiry_time": m.expiry_time.isoformat(),
        "first_joined": m.first_joined.isoformat(),
      }
    try:
      self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
      logger.error("保存会员名册失败: %s", e)

  def _load(self) -> None:
    if not self._path.exists():
      return
    try:
      raw = json.loads(self._path.read_text(encoding="utf-8"))
    except Exception as e:
      logger.error("读取会员名册失败: %s", e)
      return

    for uid, entry in raw.items():
      try:
        self._members[uid] = GuardMember(
          nickname=entry["nickname"],
          uid=uid,
          guard_level=entry["guard_level"],
          expiry_time=datetime.fromisoformat(entry["expiry_time"]),
          first_joined=datetime.fromisoformat(entry["first_joined"]),
        )
      except (KeyError, ValueError) as e:
        logger.warning("跳过无效会员条目 %s: %s", uid, e)

    before = len(self._members)
    self._cleanup()
    after = len(self._members)
    if before > after:
      logger.info("启动清理: 移除 %d 名过期会员", before - after)
    if after > 0:
      logger.info("加载会员名册: %d 名有效会员", after)

  def _cleanup(self) -> None:
    now = datetime.now(_BJT)
    expired = [uid for uid, m in self._members.items() if m.is_expired(now)]
    if expired:
      for uid in expired:
        m = self._members.pop(uid)
        logger.info("会员过期: %s (%s)", m.nickname, uid)
      self._save()

  def debug_state(self) -> dict:
    self._cleanup()
    members = []
    for m in self._members.values():
      remaining = (m.expiry_time - datetime.now(_BJT)).total_seconds()
      members.append({
        "nickname": m.nickname,
        "level": m.level_name,
        "remaining_days": round(remaining / 86400, 1),
      })
    return {
      "member_count": len(self._members),
      "members": members,
    }
