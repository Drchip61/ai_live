"""
多源风格语料下载器

从 HuggingFace datasets 和 GitHub 仓库下载语料，转换为 raw text 格式，
供 build_style_bank.py 管线处理。

Usage:
  python tools/fetch_style_corpora.py                # 下载全部源
  python tools/fetch_style_corpora.py --sources ruozhiba_punchline hitokoto
  python tools/fetch_style_corpora.py --list          # 列出可用源
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
  import requests
except ImportError:
  print("需要 requests 库: pip install requests")
  sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw_corpora"

# ============================================================
# 通用工具
# ============================================================

def _download_json(url: str, desc: str, timeout: int = 60) -> Optional[list | dict]:
  """下载 JSON 文件，返回解析后的数据"""
  print(f"  下载 {desc}...", end=" ", flush=True)
  try:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    print(f"OK ({len(resp.content) / 1024:.0f} KB)")
    return data
  except Exception as e:
    print(f"失败: {e}")
    return None


def _download_text(url: str, desc: str, timeout: int = 60) -> Optional[str]:
  """下载纯文本文件"""
  print(f"  下载 {desc}...", end=" ", flush=True)
  try:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    print(f"OK ({len(resp.content) / 1024:.0f} KB)")
    return text
  except Exception as e:
    print(f"失败: {e}")
    return None


def _download_jsonl(url: str, desc: str, timeout: int = 120) -> Optional[list[dict]]:
  """下载 JSONL 文件"""
  text = _download_text(url, desc, timeout)
  if not text:
    return None
  items = []
  for line in text.strip().split("\n"):
    line = line.strip()
    if not line:
      continue
    try:
      items.append(json.loads(line))
    except json.JSONDecodeError:
      pass
  return items


def _write_output(lines: list[str], name: str) -> Path:
  """写入输出文件"""
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  path = OUTPUT_DIR / f"{name}.txt"
  with open(path, "w", encoding="utf-8") as f:
    for line in lines:
      f.write(line.strip() + "\n")
  print(f"  → 输出: {path.relative_to(PROJECT_ROOT)} ({len(lines)} 条)")
  return path


def _load_existing_texts() -> set[str]:
  """加载现有 corpus 的文本集合，用于去重"""
  corpus_path = PROJECT_ROOT / "personas" / "dacongming" / "style_bank" / "corpus.jsonl"
  texts = set()
  if corpus_path.exists():
    for line in corpus_path.read_text(encoding="utf-8").strip().split("\n"):
      line = line.strip()
      if not line:
        continue
      try:
        entry = json.loads(line)
        texts.add(entry.get("text", "").strip())
      except json.JSONDecodeError:
        pass
  return texts


def _dedup(items: list[str], existing: set[str]) -> list[str]:
  """去重：去除已有语料和内部重复"""
  seen = set(existing)
  result = []
  for item in items:
    item = item.strip()
    if not item or len(item) < 5:
      continue
    if item in seen:
      continue
    seen.add(item)
    result.append(item)
  return result


def _clean_text(text: str) -> str:
  """基础清洗"""
  text = re.sub(r"\s+", " ", text).strip()
  text = re.sub(r"^[\d]+[.、)）]\s*", "", text)
  return text


# ============================================================
# 各数据源的下载函数
# ============================================================

def fetch_ruozhiba_punchline(existing: set[str]) -> Optional[Path]:
  """
  LooksJuicy/ruozhiba-punchline (HuggingFace)
  3.44k 条弱智吧笑话，带铺垫+笑点结构
  """
  print(f"\n{'='*50}")
  print("[ruozhiba_punchline] LooksJuicy/ruozhiba-punchline")
  print(f"{'='*50}")

  url = "https://huggingface.co/datasets/LooksJuicy/ruozhiba-punchline/resolve/main/joke_result_sample.json"
  data = _download_json(url, "joke_result_sample.json", timeout=120)
  if not data:
    return None

  lines = []
  for item in data:
    if isinstance(item, dict):
      # output 包含完整笑话，是最终版本
      text = item.get("output", "").strip()
      if not text:
        text = item.get("raw", "").strip()
      if text:
        lines.append(_clean_text(text))

  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "ruozhiba_punchline")


def fetch_ruozhiba_better(existing: set[str]) -> Optional[Path]:
  """
  FunnySaltyFish/Better-Ruozhiba (HuggingFace)
  1.48k 条精选 Q&A 对
  """
  print(f"\n{'='*50}")
  print("[ruozhiba_better] FunnySaltyFish/Better-Ruozhiba")
  print(f"{'='*50}")

  url = "https://huggingface.co/datasets/FunnySaltyFish/Better-Ruozhiba/resolve/main/ruozhiba_qa.json"
  data = _download_json(url, "ruozhiba_qa.json", timeout=120)
  if not data:
    return None

  lines = []
  for item in data:
    if isinstance(item, dict):
      q = item.get("instruction", "").strip()
      a = item.get("output", "").strip()
      if q and a:
        lines.append(_clean_text(f"{q} {a}"))
      elif q:
        lines.append(_clean_text(q))

  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "ruozhiba_better")


def fetch_kangyaba(existing: set[str]) -> Optional[Path]:
  """
  Orphanage/Baidu_Tieba_KangYaBeiGuo (HuggingFace)
  5.14k 帖，提取精华回复
  """
  print(f"\n{'='*50}")
  print("[kangyaba] Orphanage/Baidu_Tieba_KangYaBeiGuo")
  print(f"{'='*50}")

  # 该数据集用 parquet 格式，先尝试直接下载 JSON 分片
  # HuggingFace datasets viewer API
  base_url = "https://datasets-server.huggingface.co/rows"
  lines = []
  offset = 0
  page_size = 100

  print(f"  通过 API 分页拉取数据...", flush=True)
  while True:
    url = f"{base_url}?dataset=Orphanage/Baidu_Tieba_KangYaBeiGuo&config=default&split=train&offset={offset}&length={page_size}"
    try:
      resp = requests.get(url, timeout=30)
      resp.raise_for_status()
      data = resp.json()
    except Exception as e:
      if offset == 0:
        print(f"  API 拉取失败: {e}")
        return None
      break

    rows = data.get("rows", [])
    if not rows:
      break

    for row_data in rows:
      row = row_data.get("row", {})
      # 提取楼主内容（短的精华帖）
      lz = row.get("楼主内容", "").strip()
      if lz and 10 <= len(lz) <= 200:
        lines.append(_clean_text(lz))

      # 提取精华回复（短而有力的）
      replies = row.get("回复列表", [])
      if isinstance(replies, list):
        for reply in replies:
          if isinstance(reply, str):
            reply_text = reply.strip()
          elif isinstance(reply, dict):
            reply_text = reply.get("content", reply.get("text", "")).strip()
          else:
            continue
          if reply_text and 8 <= len(reply_text) <= 150:
            lines.append(_clean_text(reply_text))

    offset += len(rows)
    print(f"    已获取 {offset} 条帖子...", flush=True)
    time.sleep(0.3)  # 限速

    if offset >= 5200:
      break

  print(f"  原始提取: {len(lines)} 条")
  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "kangyaba")


def fetch_hitokoto(existing: set[str]) -> Optional[Path]:
  """
  hitokoto-osc/sentences-bundle (GitHub)
  影视名言 (h) + 抖机灵 (l) + 动画 (a) + 游戏 (c)
  """
  print(f"\n{'='*50}")
  print("[hitokoto] hitokoto-osc/sentences-bundle")
  print(f"{'='*50}")

  categories = {
    "h": "影视",
    "l": "抖机灵",
    "a": "动画",
    "c": "游戏",
  }

  lines = []
  for cat_id, cat_name in categories.items():
    url = f"https://cdn.jsdelivr.net/gh/hitokoto-osc/sentences-bundle/sentences/{cat_id}.json"
    data = _download_json(url, f"{cat_name} ({cat_id}.json)")
    if not data:
      continue

    for item in data:
      if isinstance(item, dict):
        text = item.get("hitokoto", "").strip()
        source = item.get("from", "").strip()
        if text and len(text) >= 5:
          # 影视/动漫带出处更有梗感
          if source and cat_id in ("h", "a", "c"):
            entry = f"{text} ——《{source}》"
          else:
            entry = text
          lines.append(_clean_text(entry))

  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "hitokoto")


def fetch_internet_quotes(existing: set[str]) -> Optional[Path]:
  """
  AstralSightStudios/Chinese-Internet-Quotes (GitHub)
  互联网名梗/名言
  """
  print(f"\n{'='*50}")
  print("[internet_quotes] AstralSightStudios/Chinese-Internet-Quotes")
  print(f"{'='*50}")

  # 先获取仓库文件列表
  api_url = "https://api.github.com/repos/AstralSightStudios/Chinese-Internet-Quotes/contents"
  try:
    resp = requests.get(api_url, timeout=15)
    resp.raise_for_status()
    contents = resp.json()
  except Exception as e:
    print(f"  获取仓库内容失败: {e}")
    return None

  lines = []
  # 尝试下载各 JSON/txt 文件
  for item in contents:
    name = item.get("name", "")
    if not name.endswith((".json", ".txt", ".md")):
      continue
    if name.startswith(("README", "LICENSE", ".")):
      continue
    download_url = item.get("download_url", "")
    if not download_url:
      continue

    if name.endswith(".json"):
      data = _download_json(download_url, name)
      if isinstance(data, list):
        for entry in data:
          if isinstance(entry, str) and len(entry.strip()) >= 5:
            lines.append(_clean_text(entry.strip()))
          elif isinstance(entry, dict):
            text = entry.get("text", entry.get("content", entry.get("quote", ""))).strip()
            if text and len(text) >= 5:
              lines.append(_clean_text(text))
    elif name.endswith(".txt"):
      text = _download_text(download_url, name)
      if text:
        for line in text.strip().split("\n"):
          line = line.strip()
          if line and len(line) >= 5 and not line.startswith("#"):
            lines.append(_clean_text(line))

  if not lines:
    # 备用：尝试直接下载仓库 zip
    print("  未找到可用文件，跳过")
    return None

  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "internet_quotes")


def fetch_pop_sentences(existing: set[str]) -> Optional[Path]:
  """
  Qz-Sean/Pop-Sentences (GitHub)
  发癫语录、KFC语录等
  """
  print(f"\n{'='*50}")
  print("[pop_sentences] Qz-Sean/Pop-Sentences")
  print(f"{'='*50}")

  files = {
    "psycho.json": "发癫语录",
    "kfc.json": "KFC语录",
  }

  lines = []
  for filename, desc in files.items():
    url = f"https://raw.githubusercontent.com/Qz-Sean/Pop-Sentences/main/{filename}"
    data = _download_json(url, f"{desc} ({filename})")
    if not data:
      continue

    if isinstance(data, list):
      for entry in data:
        if isinstance(entry, str):
          text = entry.strip()
        elif isinstance(entry, dict):
          text = entry.get("text", entry.get("content", entry.get("sentence", ""))).strip()
        else:
          continue
        if text and 10 <= len(text) <= 300:
          lines.append(_clean_text(text))

  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "pop_sentences")


def fetch_kfc_thursday(existing: set[str]) -> Optional[Path]:
  """
  whitescent/KFC-Crazy-Thursday (GitHub)
  疯狂星期四段子，叙事反转风格
  """
  print(f"\n{'='*50}")
  print("[kfc_thursday] whitescent/KFC-Crazy-Thursday")
  print(f"{'='*50}")

  url = "https://raw.githubusercontent.com/whitescent/KFC-Crazy-Thursday/main/kfc.json"
  data = _download_json(url, "kfc.json")
  if not data:
    return None

  lines = []
  for entry in data:
    if isinstance(entry, str):
      text = entry.strip()
    elif isinstance(entry, dict):
      text = (entry.get("text") or entry.get("content") or "").strip()
    else:
      continue
    if text and 50 <= len(text) <= 500:
      lines.append(_clean_text(text))

  lines = _dedup(lines, existing)
  if not lines:
    print("  去重后无新内容")
    return None
  return _write_output(lines, "kfc_thursday")


# ============================================================
# 源注册表
# ============================================================

SOURCES = {
  "ruozhiba_punchline": {
    "func": fetch_ruozhiba_punchline,
    "desc": "弱智吧笑话 (HF LooksJuicy/ruozhiba-punchline, 3.4k)",
    "id_prefix": "rp",
    "priority": 1,
  },
  "ruozhiba_better": {
    "func": fetch_ruozhiba_better,
    "desc": "弱智吧精选 Q&A (HF FunnySaltyFish/Better-Ruozhiba, 1.5k)",
    "id_prefix": "rb",
    "priority": 1,
  },
  "hitokoto": {
    "func": fetch_hitokoto,
    "desc": "影视/动漫/游戏名言 (GitHub hitokoto, 数千条)",
    "id_prefix": "ht",
    "priority": 2,
  },
  "kangyaba": {
    "func": fetch_kangyaba,
    "desc": "抗压吧精华 (HF Orphanage/KangYaBeiGuo, 5k帖)",
    "id_prefix": "ky",
    "priority": 3,
  },
  "internet_quotes": {
    "func": fetch_internet_quotes,
    "desc": "互联网名梗 (GitHub Chinese-Internet-Quotes)",
    "id_prefix": "iq",
    "priority": 4,
  },
  "pop_sentences": {
    "func": fetch_pop_sentences,
    "desc": "发癫/KFC语录 (GitHub Pop-Sentences)",
    "id_prefix": "ps",
    "priority": 4,
  },
  "kfc_thursday": {
    "func": fetch_kfc_thursday,
    "desc": "疯狂星期四段子 (GitHub KFC-Crazy-Thursday)",
    "id_prefix": "kf",
    "priority": 5,
  },
}


# ============================================================
# 入口
# ============================================================

def main():
  parser = argparse.ArgumentParser(description="多源风格语料下载器")
  parser.add_argument(
    "--sources", nargs="+",
    help="指定要下载的源（默认全部）",
  )
  parser.add_argument(
    "--list", action="store_true",
    help="列出所有可用数据源",
  )
  parser.add_argument(
    "--no-dedup", action="store_true",
    help="跳过与现有 corpus 的去重",
  )
  args = parser.parse_args()

  if args.list:
    print("可用数据源:")
    for name, info in sorted(SOURCES.items(), key=lambda x: x[1]["priority"]):
      print(f"  {name:25s} P{info['priority']}  {info['desc']}")
    return

  sources_to_fetch = args.sources or list(SOURCES.keys())
  invalid = [s for s in sources_to_fetch if s not in SOURCES]
  if invalid:
    print(f"未知数据源: {', '.join(invalid)}")
    print(f"可用: {', '.join(SOURCES.keys())}")
    sys.exit(1)

  existing = set() if args.no_dedup else _load_existing_texts()
  if existing:
    print(f"已加载 {len(existing)} 条现有语料用于去重")

  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  results: dict[str, Optional[Path]] = {}

  sorted_sources = sorted(sources_to_fetch, key=lambda s: SOURCES[s]["priority"])
  for name in sorted_sources:
    info = SOURCES[name]
    path = info["func"](existing)
    results[name] = path
    # 把新下载的也加入去重集合
    if path and path.exists():
      for line in path.read_text(encoding="utf-8").strip().split("\n"):
        existing.add(line.strip())

  print(f"\n{'='*50}")
  print("下载汇总")
  print(f"{'='*50}")
  for name, path in results.items():
    if path:
      count = len(path.read_text(encoding="utf-8").strip().split("\n"))
      prefix = SOURCES[name]["id_prefix"]
      print(f"  OK  {name:25s} {count:5d} 条  (id_prefix={prefix})")
    else:
      print(f"  --  {name:25s} 跳过或失败")

  print(f"\n输出目录: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
  print(
    "下一步: 对每个 .txt 文件运行 build_style_bank.py --append，例如:\n"
    "  python tools/build_style_bank.py data/raw_corpora/ruozhiba_punchline.txt "
    "-o personas/dacongming/style_bank/corpus.jsonl --append --id-prefix rp --source ruozhiba"
  )


if __name__ == "__main__":
  main()
