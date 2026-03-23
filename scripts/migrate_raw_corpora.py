"""
将 data/raw_corpora/*.txt 转换为 corpus_store.json (CorpusEntry 格式)

用法：
  python scripts/migrate_raw_corpora.py
  python scripts/migrate_raw_corpora.py --dry-run   # 只统计不写入
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_corpora"
OUTPUT_PATH = (
  PROJECT_ROOT / "data" / "memory_store" / "structured" / "corpus_store.json"
)

MIN_TEXT_LENGTH = 5

SOURCE_CONFIG = {
  "hitokoto.txt": {
    "kind": "classic_quote",
    "style_tags": ["感性", "文艺"],
    "scene_tags": ["冷场", "互动"],
    "quality_score": 0.7,
  },
  "kfc_thursday.txt": {
    "kind": "kfc_meme",
    "style_tags": ["搞笑", "反转"],
    "scene_tags": ["互动", "冷场"],
    "quality_score": 0.7,
  },
  "ruozhiba_punchline_sampled.txt": {
    "kind": "absurdist_joke",
    "style_tags": ["搞笑", "脑洞"],
    "scene_tags": ["互动", "冷场"],
    "quality_score": 0.7,
  },
  "ruozhiba_better_sampled.txt": {
    "kind": "absurdist_qa",
    "style_tags": ["搞笑", "脑洞", "一本正经"],
    "scene_tags": ["互动"],
    "quality_score": 0.7,
  },
  "internet_quotes.txt": {
    "kind": "internet_meme",
    "style_tags": ["搞笑", "阴阳"],
    "scene_tags": ["互动", "吐槽"],
    "quality_score": 0.5,
  },
  "kangyaba_sampled.txt": {
    "kind": "sarcasm",
    "style_tags": ["阴阳", "吐槽"],
    "scene_tags": ["吐槽", "互动"],
    "quality_score": 0.6,
  },
  "pop_sentences_sampled.txt": {
    "kind": "pop_culture",
    "style_tags": ["搞笑", "流行"],
    "scene_tags": ["互动", "冷场"],
    "quality_score": 0.7,
  },
}


def migrate(dry_run: bool = False) -> None:
  entries: list[dict] = []
  now = datetime.now().isoformat()
  stats: dict[str, int] = {}

  for filename, config in SOURCE_CONFIG.items():
    filepath = RAW_DIR / filename
    if not filepath.exists():
      print(f"  [跳过] 文件不存在: {filename}")
      continue

    lines = filepath.read_text(encoding="utf-8").splitlines()
    counter = 0
    for line in lines:
      text = line.strip()
      if len(text) < MIN_TEXT_LENGTH:
        continue
      counter += 1
      corpus_id = f"{config['kind']}_{counter:04d}"
      entries.append({
        "corpus_id": corpus_id,
        "kind": config["kind"],
        "text": text,
        "style_tags": config["style_tags"],
        "scene_tags": config["scene_tags"],
        "constraints": [],
        "quality_score": config["quality_score"],
        "source": filename,
        "enabled": True,
        "updated_at": now,
      })
    stats[filename] = counter
    print(f"  {filename}: {counter} 条")

  print(f"\n总计: {len(entries)} 条语料")

  if dry_run:
    print("(dry-run 模式，未写入文件)")
    return

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  OUTPUT_PATH.write_text(
    json.dumps(entries, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"已写入: {OUTPUT_PATH}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="迁移 raw_corpora 到 corpus_store.json")
  parser.add_argument("--dry-run", action="store_true", help="只统计不写入")
  args = parser.parse_args()
  migrate(dry_run=args.dry_run)
