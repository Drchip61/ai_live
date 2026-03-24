"""
Prompt 运行时 contract

把检索层和组合层之间传递的数据结构统一起来，
避免继续用裸字符串和松散 tuple 传递语义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


ContextTrust = Literal["trusted", "untrusted"]


@dataclass(frozen=True)
class ContextBlock:
  """单个上下文块，带来源和信任级别。"""

  source: str
  text: str
  trust: ContextTrust = "untrusted"
  title: str = ""
  query_used: str = ""

  def render(self) -> str:
    body = str(self.text or "").strip()
    if not body:
      return ""
    title = str(self.title or "").strip()
    if title:
      return f"{title}\n{body}"
    return body


@dataclass(frozen=True)
class RetrievedContextBundle:
  """Retriever 层输出：结构化上下文块 + 检索元信息。"""

  blocks: tuple[ContextBlock, ...] = field(default_factory=tuple)
  retrieval_query: str = ""
  writeback_input: str = ""
  viewer_ids: tuple[str, ...] = field(default_factory=tuple)

  def by_trust(self, trust: ContextTrust) -> tuple[ContextBlock, ...]:
    return tuple(block for block in self.blocks if block.trust == trust)

  def render_trusted_text(self) -> str:
    return "\n\n".join(
      text for text in (block.render() for block in self.by_trust("trusted")) if text
    )

  def render_untrusted_text(self) -> str:
    return "\n\n".join(
      text for text in (block.render() for block in self.by_trust("untrusted")) if text
    )

  def debug_view(self) -> dict:
    return {
      "retrieval_query": self.retrieval_query,
      "writeback_input": self.writeback_input,
      "viewer_ids": list(self.viewer_ids),
      "trusted_sources": [block.source for block in self.by_trust("trusted")],
      "untrusted_sources": [block.source for block in self.by_trust("untrusted")],
    }


@dataclass(frozen=True)
class ModelInvocation:
  """Composer 层输出：最终要交给 pipeline 的模型调用载荷。"""

  user_prompt: str
  images: Optional[list[str]] = None
  trusted_context: str = ""
  untrusted_context: str = ""
  response_style: str = "normal"
  route_kind: str = "chat"
