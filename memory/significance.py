"""
significance 评分计算
独立函数，方便日后修改算法
"""


def decay_significance(
  current: float,
  coefficient: float,
) -> float:
  """
  衰减 significance（未被取用时调用）

  算法：significance *= coefficient

  Args:
    current: 当前 significance 值
    coefficient: 衰减系数（如 0.95）

  Returns:
    衰减后的 significance，保留三位小数
  """
  return round(current * coefficient, 3)


def boost_significance(current: float) -> float:
  """
  提升 significance（被 RAG 取用时调用）

  算法：与 1 之间的差值缩减一半
    new = current + (1 - current) / 2

  Args:
    current: 当前 significance 值

  Returns:
    提升后的 significance，保留三位小数
  """
  return round(current + (1.0 - current) / 2.0, 3)


def initial_significance() -> float:
  """
  新记忆的初始 significance

  Returns:
    初始值 0.500
  """
  return 0.500
