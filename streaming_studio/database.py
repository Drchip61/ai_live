"""
数据库模块
使用 SQLite 存储弹幕和回复记录
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Comment, StreamerResponse


class CommentDatabase:
  """
  弹幕数据库
  使用 SQLite 存储弹幕和主播回复
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

    # 内存模式需要保持单一连接（关闭即销毁）
    if self._in_memory:
      self._shared_conn = sqlite3.connect(
        ":memory:", check_same_thread=False,
      )
    else:
      self._shared_conn = None

    self._init_database()

  def _get_connection(self) -> sqlite3.Connection:
    """获取数据库连接"""
    if self._shared_conn is not None:
      return self._shared_conn
    return sqlite3.connect(self._db_path)

  def _init_database(self) -> None:
    """初始化数据库表"""
    with self._get_connection() as conn:
      cursor = conn.cursor()

      # 创建弹幕表
      cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
          id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          nickname TEXT NOT NULL,
          content TEXT NOT NULL,
          timestamp TEXT NOT NULL
        )
      """)

      # 创建回复表
      cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
          id TEXT PRIMARY KEY,
          content TEXT NOT NULL,
          reply_to TEXT NOT NULL,
          timestamp TEXT NOT NULL
        )
      """)

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
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        INSERT OR REPLACE INTO comments (id, user_id, nickname, content, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
          comment.id,
          comment.user_id,
          comment.nickname,
          comment.content,
          comment.timestamp.isoformat()
        )
      )
      conn.commit()

  def save_response(self, response: StreamerResponse) -> None:
    """
    保存主播回复

    Args:
      response: 回复对象
    """
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        INSERT OR REPLACE INTO responses (id, content, reply_to, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
          response.id,
          response.content,
          json.dumps(list(response.reply_to)),
          response.timestamp.isoformat()
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
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        "SELECT id, user_id, nickname, content, timestamp FROM comments WHERE id = ?",
        (comment_id,)
      )
      row = cursor.fetchone()

      if row is None:
        return None

      return Comment(
        id=row[0],
        user_id=row[1],
        nickname=row[2],
        content=row[3],
        timestamp=datetime.fromisoformat(row[4])
      )

  def get_recent_comments(self, limit: int = 10) -> list[Comment]:
    """
    获取最近的弹幕

    Args:
      limit: 返回数量限制

    Returns:
      弹幕列表，按时间倒序
    """
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        SELECT id, user_id, nickname, content, timestamp
        FROM comments
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
      )
      rows = cursor.fetchall()

      return [
        Comment(
          id=row[0],
          user_id=row[1],
          nickname=row[2],
          content=row[3],
          timestamp=datetime.fromisoformat(row[4])
        )
        for row in rows
      ]

  def get_recent_responses(self, limit: int = 10) -> list[StreamerResponse]:
    """
    获取最近的回复

    Args:
      limit: 返回数量限制

    Returns:
      回复列表，按时间倒序
    """
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        """
        SELECT id, content, reply_to, timestamp
        FROM responses
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,)
      )
      rows = cursor.fetchall()

      return [
        StreamerResponse(
          id=row[0],
          content=row[1],
          reply_to=tuple(json.loads(row[2])),
          timestamp=datetime.fromisoformat(row[3])
        )
        for row in rows
      ]

  def get_comment_count(self) -> int:
    """获取弹幕总数"""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM comments")
      return cursor.fetchone()[0]

  def get_response_count(self) -> int:
    """获取回复总数"""
    with self._get_connection() as conn:
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
    with self._get_connection() as conn:
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
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
        "UPDATE streaming_sessions SET end_time = ? WHERE session_id = ?",
        (datetime.now().isoformat(), session_id),
      )
      conn.commit()
