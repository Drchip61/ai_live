"""
语料清洗工具：移除含平台名的元帖子 + 清洗来源后缀

Usage:
  python tools/clean_corpus.py personas/dacongming/style_bank/corpus.jsonl
  python tools/clean_corpus.py corpus.jsonl --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 来源后缀：如 "——来自 爱贴吧 xxx 客户端"
_SOURCE_SUFFIX_RE = re.compile(
  r"\s*[—\-–]+\s*来自\s*.{1,30}(?:客户端|App|app)\s*$"
)

# 直接移除：文本包含具体社区/贴吧名（风险大于保留价值）
_COMMUNITY_BLACKLIST = re.compile(r"弱智吧|抗压吧|气功吧")

# 二级检测：通用"贴吧"出现在社区语境中
_TIEBA_META = re.compile(r"贴吧")
_META_SIGNALS = re.compile(
  r"吧友|病友|吧主|精品贴|发帖|新人报到|报到处|驻.*吧"
  r"|感动.*人物评选|资格测试|经验加三|回复.*火"
)


def _is_meta_post(text: str) -> bool:
  """判断是否含社区来源信息（应删除）"""
  if _COMMUNITY_BLACKLIST.search(text):
    return True
  if _TIEBA_META.search(text) and _META_SIGNALS.search(text):
    return True
  return False


def _clean_source_suffix(text: str) -> str:
  """移除尾部来源标记"""
  return _SOURCE_SUFFIX_RE.sub("", text)


def clean_corpus(input_path: Path, dry_run: bool = False) -> None:
  lines = input_path.read_text(encoding="utf-8").strip().split("\n")

  kept: list[str] = []
  removed_meta: list[dict] = []
  cleaned_suffix: list[dict] = []

  for line in lines:
    line = line.strip()
    if not line:
      continue
    try:
      entry = json.loads(line)
    except json.JSONDecodeError:
      kept.append(line)
      continue

    text = entry.get("text", "")

    if _is_meta_post(text):
      removed_meta.append(entry)
      continue

    new_text = _clean_source_suffix(text)
    if new_text != text:
      cleaned_suffix.append({"id": entry.get("id"), "before": text, "after": new_text})
      entry["text"] = new_text

    kept.append(json.dumps(entry, ensure_ascii=False))

  print(f"总条目: {len(lines)}")
  print(f"删除元帖子: {len(removed_meta)} 条")
  print(f"清洗来源后缀: {len(cleaned_suffix)} 条")
  print(f"保留: {len(kept)} 条")

  if removed_meta:
    print("\n--- 删除的元帖子 ---")
    for e in removed_meta:
      preview = e.get("text", "")[:80]
      print(f"  [{e.get('id', '?')}] {preview}...")

  if cleaned_suffix:
    print("\n--- 清洗的来源后缀 ---")
    for c in cleaned_suffix:
      print(f"  [{c['id']}] ...{c['before'][-40:]}  →  ...{c['after'][-40:]}")

  if dry_run:
    print("\n(dry-run 模式，未写入文件)")
  else:
    input_path.write_text(
      "\n".join(kept) + "\n", encoding="utf-8"
    )
    print(f"\n已写入 → {input_path}")


def main():
  parser = argparse.ArgumentParser(description="语料清洗：移除元帖子 + 清洗来源后缀")
  parser.add_argument("input", help="corpus.jsonl 路径")
  parser.add_argument("--dry-run", action="store_true", help="只预览不写入")
  args = parser.parse_args()

  path = Path(args.input)
  if not path.exists():
    print(f"错误: 文件不存在 → {path}")
    sys.exit(1)

  clean_corpus(path, dry_run=args.dry_run)


if __name__ == "__main__":
  main()
