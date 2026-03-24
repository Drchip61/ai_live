"""
数据库模块
使用 SQLite 存储弹幕和回复记录
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Comment, StreamerResponse


class CommentDatabase:
  """
  弹幕数据库
  使用 SQLite 存储弹幕和主播回复

  所有写操作通过 threading.Lock 序列化，
  共享单一连接（check_same_thread=False）以避免并发写入导致 SQLITE_MISUSE。
  """

  def __init__(self, db_path: Optional[str] = None):
    """
    初始化数据库

    Args:
      db_path: 数据库路径。
        - None: 默认文件路径 data/comments.db
        - ":memory:": 纯内存数据库（进程结束即销毁）
        - 其他字符串: 指定文件路径
    """
    self._in_memory = (db_path == ":memory:")

    if db_path is None:
      project_root = Path(__file__).parent.parent
      data_dir = project_root / "data"
      data_dir.mkdir(exist_ok=True)
      db_path = str(data_dir / "comments.db")

    self._db_path = db_path
    self._lock = threading.Lock()

    self._shared_conn = sqlite3.connect(
      ":memory:" if self._in_memory else self._db_path,
      check_same_thread=False,
    )

    self._init_database()

  def _get_connection(self) -> sqlite3.Connection:
    """获取数据库连接（始终返回共享连接）"""
    return self._shared_conn

  def _init_database(self) -> None:
    """初始化数据库表"""
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()

      # 创建弹幕表
      cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          nickname TEXT NOT NULL,
          content TEXT NOT NULL,
          timestamp TEXT NOT NULL,
          event_type TEXT NOT NULL DEFAULT 'danmaku',
          gift_name TEXT DEFAULT '',
          gift_num INTEGER DEFAULT 0,
          price REAL DEFAULT 0.0,
          guard_level INTEGER DEFAULT 0
        )
      """)

      # 旧表升级：逐列尝试添加，已存在则跳过
      for col, typedef in [
        ("event_type", "TEXT NOT NULL DEFAULT 'danmaku'"),
        ("gift_name", "TEXT DEFAULT ''"),
        ("gift_num", "INTEGER DEFAULT 0"),
        ("price", "REAL DEFAULT 0.0"),
        ("guard_level", "INTEGER DEFAULT 0"),
      ]:
        try:
          cursor.execute(f"ALTER TABLE comments ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
          pass

      # 创建回复表
      cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
          id TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          reply_to TEXT NOT NULL,
          reply_target_text TEXT DEFAULT '',
          nickname TEXT DEFAULT '',
          timestamp TEXT NOT NULL,
          response_style TEXT NOT NULL DEFAULT 'normal',
          controller_trace_json TEXT DEFAULT '',
          timing_trace_json TEXT DEFAULT ''
        )
      """)

      for col, typedef in [
        ("reply_target_text", "TEXT DEFAULT ''"),
        ("nickname", "TEXT DEFAULT ''"),
        ("response_style", "TEXT NOT NULL DEFAULT 'normal'"),
        ("controller_trace_json", "TEXT DEFAULT ''"),
        ("timing_trace_json", "TEXT DEFAULT ''"),
      ]:
        try:
          cursor.execute(f"ALTER TABLE responses ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
          pass

      # 创建直播会话表
      cursor.execute("""
        CREATE TABLE IF NOT EXISTS streaming_sessions (
          session_id TEXT PRIMARY KEY,
          persona TEXT NOT NULL,
          start_time TEXT NOT NULL,
          end_time TEXT
        )
      """)

      # 创建索引
      cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_comments_timestamp
        ON comments(timestamp)
      """)
      cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_responses_timestamp
        ON responses(timestamp)
      """)

      conn.commit()

  def save_comment(self, comment: Comment) -> None:
    """
    保存弹幕

    Args:
      comment: 弹幕对象
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        INSERT OR REPLACE INTO comments
          (id, user_id, nickname, content, timestamp,
           event_type, gift_name, gift_num, price, guard_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          comment.id,
          comment.user_id,
          comment.nickname,
          comment.content,
          comment.timestamp.isoformat(),
          comment.event_type.value,
          comment.gift_name,
          comment.gift_num,
          comment.price,
          comment.guard_level,
        )
      )
      conn.commit()

  def save_response(self, response: StreamerResponse) -> None:
    """
    保存主播回复

    Args:
      response: 回复对象
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        INSERT OR REPLACE INTO responses
          (id, content, reply_to, reply_target_text, nickname, timestamp,
           response_style, controller_trace_json, timing_trace_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
          response.id,
          response.content,
          json.dumps(list(response.reply_to)),
          response.reply_target_text,
          response.nickname,
          response.timestamp.isoformat(),
          response.response_style,
          json.dumps(response.controller_trace, ensure_ascii=False) if response.controller_trace is not None else "",
          json.dumps(response.timing_trace, ensure_ascii=False) if response.timing_trace is not None else "",
        )
      )
      conn.commit()

  def get_comment(self, comment_id: str) -> Optional[Comment]:
    """
    根据ID获取弹幕

    Args:
      comment_id: 弹幕ID

    Returns:
      弹幕对象，不存在则返回None
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """SELECT id, user_id, nickname, content, timestamp,
                  event_type, gift_name, gift_num, price, guard_level
           FROM comments WHERE id = ?""",
        (comment_id,)
      )
      row = cursor.fetchone()

      if row is None:
        return None

      from .models import EventType
      try:
        event_type = EventType(row[5])
      except (ValueError, IndexError):
        event_type = EventType.DANMAKU

      return Comment(
        id=row[0],
        user_id=row[1],
        nickname=row[2],
        content=row[3],
        timestamp=datetime.fromisoformat(row[4]),
        event_type=event_type,
        gift_name=row[6] or "",
        gift_num=row[7] or 0,
        price=row[8] or 0.0,
        guard_level=row[9] or 0,
      )

  def get_recent_comments(self, limit: int = 10) -> list[Comment]:
    """
    获取最近的弹幕

    Args:
      limit: 返回数量限制

    Returns:
      弹幕列表，按时间倒序
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        SELECT id, user_id, nickname, content, timestamp,
               event_type, gift_name, gift_num, price, guard_level
        FROM comments
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
      )
      rows = cursor.fetchall()

      from .models import EventType
      results = []
      for row in rows:
        try:
          event_type = EventType(row[5])
        except (ValueError, IndexError):
          event_type = EventType.DANMAKU
        results.append(Comment(
          id=row[0],
          user_id=row[1],
          nickname=row[2],
          content=row[3],
          timestamp=datetime.fromisoformat(row[4]),
          event_type=event_type,
          gift_name=row[6] or "",
          gift_num=row[7] or 0,
          price=row[8] or 0.0,
          guard_level=row[9] or 0,
        ))
      return results

  def get_recent_responses(self, limit: int = 10) -> list[StreamerResponse]:
    """
    获取最近的回复

    Args:
      limit: 返回数量限制

    Returns:
      回复列表，按时间倒序
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        SELECT id, content, reply_to, reply_target_text, nickname, timestamp,
               response_style, controller_trace_json, timing_trace_json
        FROM responses
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
      )
      rows = cursor.fetchall()

      def _loads_optional_json(text: str) -> Optional[dict]:
        if not text:
          return None
        try:
          data = json.loads(text)
        except json.JSONDecodeError:
          return None
        return data if isinstance(data, dict) else None

      return [
        StreamerResponse(
          id=row[0],
          content=row[1],
          reply_to=tuple(json.loads(row[2])),
          reply_target_text=row[3] or "",
          nickname=row[4] or "",
          timestamp=datetime.fromisoformat(row[5]),
          response_style=row[6] or "normal",
          controller_trace=_loads_optional_json(row[7]),
          timing_trace=_loads_optional_json(row[8]),
        )
        for row in rows
      ]

  def get_comment_count(self) -> int:
    """获取弹幕总数"""
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM comments")
      return cursor.fetchone()[0]

  def get_response_count(self) -> int:
    """获取回复总数"""
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM responses")
      return cursor.fetchone()[0]

  def create_session(
    self,
    session_id: str,
    persona: str,
  ) -> None:
    """
    创建直播会话记录（开播时调用）

    Args:
      session_id: 会话 ID
      persona: 角色名称
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        INSERT OR REPLACE INTO streaming_sessions
          (session_id, persona, start_time) VALUES (?, ?, ?)
        """,
        (session_id, persona, datetime.now().isoformat()),
      )
      conn.commit()

  def end_session(self, session_id: str) -> None:
    """
    结束直播会话（下播时调用）

    Args:
      session_id: 会话 ID
    """
    with self._lock, self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        "UPDATE streaming_sessions SET end_time = ? WHERE session_id = ?",
        (datetime.now().isoformat(), session_id),
      )
      conn.commit()
