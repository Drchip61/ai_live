"""
提示词加载器
负责加载通用提示词，并组装完整的系统提示词
"""

import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在路径中
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

from personas import PersonaLoader


class PromptLoader:
  """
  提示词加载器

  管理通用提示词文件（如 base_instruction），
  并通过 PersonaLoader 获取角色信息，组装完整的系统提示词。
  """

  def __init__(self, prompts_dir: Optional[Path] = None):
    """
    初始化提示词加载器

    Args:
      prompts_dir: 提示词文件所在目录，默认为当前模块所在目录
    """
    if prompts_dir is None:
      self.prompts_dir = Path(__file__).parent
    else:
      self.prompts_dir = Path(prompts_dir)

    self._persona_loader = PersonaLoader()

  def load(self, filename: str) -> str:
    """
    加载指定的提示词文件

    Args:
      filename: 文件名或相对路径（如 "topic/single_classify.txt"）

    Returns:
      文件内容字符串

    Raises:
      FileNotFoundError: 文件不存在时抛出
    """
    file_path = self.prompts_dir / filename
    if not file_path.exists():
      raise FileNotFoundError(f"提示词文件不存在: {file_path}")
    return file_path.read_text(encoding="utf-8")

  def load_template(self, path: str, **kwargs: str) -> str:
    """
    加载 txt 模板并填充变量

    Args:
      path: 模板文件相对路径
      **kwargs: 模板变量

    Returns:
      填充后的字符串
    """
    raw = self.load(path)
    if kwargs:
      return raw.format(**kwargs)
    return raw

  def load_headers(self, path: str) -> dict[str, str]:
    """
    加载 key=value 格式的标题映射文件

    文件格式：每行一个 key=value，空行和 # 开头的行跳过

    Args:
      path: 文件相对路径

    Returns:
      dict[key, value]
    """
    raw = self.load(path)
    headers: dict[str, str] = {}
    for line in raw.splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      if "=" in line:
        key, value = line.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers

  def get_base_instruction(self) -> str:
    """
    获取基础指令

    Returns:
      基础指令内容
    """
    return self.load("base_instruction.txt")

  def get_full_system_prompt(self, persona: str) -> str:
    """
    获取完整的系统提示词（基础指令 + 角色提示词）

    通过 PersonaLoader 获取角色专属提示词，与基础指令拼接。

    Args:
      persona: 角色名称

    Returns:
      完整的系统提示词
    """
    base = self.get_base_instruction()
    persona_prompt = self._persona_loader.get_system_prompt(persona)
    return f"{base}\n\n{persona_prompt}"

  def list_personas(self) -> list[str]:
    """
    列出所有可用角色（委托给 PersonaLoader）

    Returns:
      角色名称列表
    """
    return self._persona_loader.list_personas()
