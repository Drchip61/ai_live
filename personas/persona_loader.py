"""
人格加载器
从 personas/{角色名}/ 目录加载角色相关的各类信息
"""

from pathlib import Path
from typing import Optional


class PersonaLoader:
  """
  人格加载器

  自动发现 personas/ 下的角色子目录，加载角色相关文件。
  每个角色一个文件夹，当前支持 system_prompt.txt，未来可扩展更多分类。

  目录结构：
    personas/
    ├── karin/
    │   └── system_prompt.txt
    ├── sage/
    │   └── system_prompt.txt
    └── kuro/
        └── system_prompt.txt
  """

  def __init__(self, personas_dir: Optional[Path] = None):
    """
    初始化人格加载器

    Args:
      personas_dir: 人格文件根目录，默认为当前模块所在目录
    """
    if personas_dir is None:
      self._dir = Path(__file__).parent
    else:
      self._dir = Path(personas_dir)

  def list_personas(self) -> list[str]:
    """
    列出所有可用角色（自动发现子目录）

    Returns:
      角色名称列表（排序后）
    """
    return sorted([
      d.name for d in self._dir.iterdir()
      if d.is_dir() and not d.name.startswith(("_", "."))
    ])

  def load(self, persona: str, filename: str) -> str:
    """
    加载角色目录下的任意文件

    Args:
      persona: 角色名称
      filename: 文件名（含扩展名）

    Returns:
      文件内容

    Raises:
      ValueError: 角色不存在
      FileNotFoundError: 文件不存在
    """
    persona = persona.lower()
    persona_dir = self._dir / persona
    if not persona_dir.is_dir():
      raise ValueError(
        f"角色不存在: {persona}，可用角色: {self.list_personas()}"
      )

    file_path = persona_dir / filename
    if not file_path.exists():
      raise FileNotFoundError(
        f"角色文件不存在: {persona}/{filename}"
      )
    return file_path.read_text(encoding="utf-8")

  def get_system_prompt(self, persona: str) -> str:
    """
    获取角色专属的系统提示词

    Args:
      persona: 角色名称

    Returns:
      角色系统提示词内容
    """
    return self.load(persona, "system_prompt.txt")
