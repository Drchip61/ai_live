"""
风格参考库语料处理管线

读取原始文本 → 批量 LLM 评分/过滤/分类/标注 → 输出 corpus.jsonl

Usage:
  python tools/build_style_bank.py input.txt -o personas/dacongming/style_bank/corpus.jsonl
  python tools/build_style_bank.py input.txt -o corpus.jsonl --model anthropic --min-score 4
  python tools/build_style_bank.py input.txt -o corpus.jsonl --adapt --append
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_wrapper.model_provider import ModelProvider, ModelType

# ============================================================
# Prompts
# ============================================================

SCORE_SYSTEM = """\
你是弱智吧/抽象梗内容质量评审。评估每条内容的质量。

评分标准（1-5）：
5 - 经典：逻辑结构精妙 + 前提荒诞但推理有效 + 结论出人意料，先愣住再爆笑
4 - 优秀：有创意有趣味，逻辑链条有意思，但不够惊艳
3 - 尚可：有弱智吧风格，但比较普通或类似内容太多
2 - 一般：创意不足、逻辑不通、或只是普通段子
1 - 垃圾：无聊/低俗/纯灌水/跟弱智吧风格无关

重点关注：
- 「一本正经胡说八道」的反差感
- 逻辑推理的完整性（即使前提荒诞）
- 出人意料的角度和结论
- 能否在直播间口语化表达"""

SCORE_USER_TEMPLATE = """\
请评估以下 {count} 条内容的质量。

{items}

对每条输出 JSON 数组：[{{"id": 1, "score": 5, "brief": "一句话理由"}}, ...]
只输出 JSON 数组，不要输出其他内容。"""

CLASSIFY_SYSTEM = """\
你是直播间风格语料分类专家。为每条内容标注类别和使用场景。

category（内容类型）:
- classic_question: 经典荒诞假设问题（"如果蚊子吸了运动员的血..."）
- reasoning_chain: 完整推理链，从荒诞前提推导结论（"首先...其次...因此..."）
- comment_reaction: 适合回应别人发言（"你说你来了，但来的定义是什么"）
- scene_reaction: 适合评论看到的画面（"等等，这个角色走路姿势有逻辑漏洞"）
- ice_breaker: 适合冷场时主动抛出（"既然没人说话，我提一个问题..."）
- comeback: 适合被质疑/攻击时回击（"你的反驳缺乏逻辑结构"）

situation（使用场景）:
- proactive: 主动发言（冷场、开场、自言自语）
- react_comment: 回应观众弹幕
- react_scene: 评论直播画面
- comeback: 被观众怼时的回击
- any: 通用，多种场景都能用

tags: 2-4 个语义标签（如 "动物"、"物理"、"哲学"、"生活" 等）"""

CLASSIFY_USER_TEMPLATE = """\
请为以下 {count} 条内容分类。

{items}

对每条输出 JSON 数组：
[{{"id": 1, "category": "classic_question", "situation": "proactive", "tags": ["动物", "运动"]}}, ...]
只输出 JSON 数组，不要输出其他内容。"""

ADAPT_SYSTEM = """\
你是直播间口语化改写专家。将弱智吧风格内容改写为更适合直播口语表达的版本。

改写规则：
- 保留核心逻辑和荒诞点，只调整表达方式
- 书面语 → 口语（"因此" → "所以"、"然而" → "但是"）
- 过长的内容适当缩短（直播间一次不能说太长）
- 保持一本正经的语气，不要加笑声或可爱语气词
- 原文已经很口语化的，保持不变"""

ADAPT_USER_TEMPLATE = """\
请改写以下 {count} 条内容为直播间口语版本。

{items}

对每条输出 JSON 数组：[{{"id": 1, "text": "改写后的文本"}}, ...]
只输出 JSON 数组，不要输出其他内容。"""


# ============================================================
# 文本解析
# ============================================================

def parse_raw_text(text: str) -> list[str]:
  """
  将原始文本拆分为独立条目。
  自动检测分隔方式：空行分隔（多段落）或逐行（短帖列表）。
  """
  lines = text.strip().split("\n")
  has_paragraphs = False
  consecutive_nonempty = 0
  for line in lines:
    if line.strip() == "":
      if consecutive_nonempty > 1:
        has_paragraphs = True
        break
      consecutive_nonempty = 0
    else:
      consecutive_nonempty += 1

  if has_paragraphs:
    items = []
    current: list[str] = []
    for line in lines:
      if line.strip() == "":
        if current:
          items.append("\n".join(current))
          current = []
      else:
        current.append(line)
    if current:
      items.append("\n".join(current))
  else:
    items = [line for line in lines if line.strip()]

  cleaned = []
  for item in items:
    item = _clean_item(item)
    if item and len(item) >= 5:
      cleaned.append(item)
  return cleaned


def _clean_item(text: str) -> str:
  """清洗单条内容：去编号、去 markdown 格式等"""
  text = text.strip()
  text = re.sub(r"^[\d]+[.、)）]\s*", "", text)
  text = re.sub(r"^\([\d]+\)\s*", "", text)
  text = re.sub(r"^第[\d一二三四五六七八九十]+[条个]\s*[：:.]?\s*", "", text)
  text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
  text = re.sub(r"__(.+?)__", r"\1", text)
  text = re.sub(r"^#+\s*", "", text)
  text = re.sub(r"^[-*]\s+", "", text)
  text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
  return text.strip()


# ============================================================
# LLM 批量调用
# ============================================================

def _batch(items: list, size: int) -> list[list]:
  return [items[i:i + size] for i in range(0, len(items), size)]


def _format_numbered(texts: list[str], offset: int = 0) -> str:
  return "\n".join(f"{i}. {t}" for i, t in enumerate(texts, start=offset + 1))


def _parse_json_array(text: str) -> list:
  """容错解析 LLM 返回的 JSON 数组"""
  text = text.strip()
  if "```" in text:
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
      text = m.group(1).strip()

  start, end = text.find("["), text.rfind("]")
  if start != -1 and end != -1 and end > start:
    text = text[start:end + 1]

  try:
    return json.loads(text)
  except json.JSONDecodeError:
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)
    try:
      return json.loads(text)
    except json.JSONDecodeError:
      return []


def _call_llm(model, system: str, user: str, retries: int = 3) -> str:
  for attempt in range(retries):
    try:
      resp = model.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
      ])
      return resp.content
    except Exception as e:
      if attempt < retries - 1:
        wait = 2 ** (attempt + 1)
        print(f"  [重试] {e}，{wait}s 后重试...", flush=True)
        time.sleep(wait)
      else:
        print(f"  [失败] {e}", flush=True)
        return "[]"


# ============================================================
# 管线三步
# ============================================================

def step_score(model, items: list[str], batch_size: int = 15) -> dict[int, dict]:
  """Step 1: 质量评分"""
  print(f"\n{'='*50}")
  print(f"[Step 1] 质量评分（{len(items)} 条，每批 {batch_size}）")
  print(f"{'='*50}")

  scores: dict[int, dict] = {}
  batches = _batch(list(enumerate(items)), batch_size)

  for bi, chunk in enumerate(batches):
    texts = [t for _, t in chunk]
    indices = [idx for idx, _ in chunk]
    prompt = SCORE_USER_TEMPLATE.format(
      count=len(chunk),
      items=_format_numbered(texts),
    )

    print(f"  批次 {bi+1}/{len(batches)} ({len(chunk)} 条)...", end=" ", flush=True)
    raw = _call_llm(model, SCORE_SYSTEM, prompt)
    results = _parse_json_array(raw)

    ok = 0
    for r in results:
      if isinstance(r, dict) and "id" in r and "score" in r:
        pi = r["id"] - 1
        if 0 <= pi < len(chunk):
          scores[indices[pi]] = r
          ok += 1
    print(f"完成 {ok}/{len(chunk)}")

    for orig_idx, _ in chunk:
      if orig_idx not in scores:
        scores[orig_idx] = {"id": orig_idx, "score": 3, "brief": "未评分，默认保留"}

  return scores


def step_classify(model, items: list[tuple[int, str]], batch_size: int = 12) -> dict[int, dict]:
  """Step 2: 分类 + 情境标注"""
  print(f"\n{'='*50}")
  print(f"[Step 2] 分类标注（{len(items)} 条，每批 {batch_size}）")
  print(f"{'='*50}")

  classes: dict[int, dict] = {}
  batches = _batch(items, batch_size)

  for bi, chunk in enumerate(batches):
    texts = [t for _, t in chunk]
    indices = [idx for idx, _ in chunk]
    prompt = CLASSIFY_USER_TEMPLATE.format(
      count=len(chunk),
      items=_format_numbered(texts),
    )

    print(f"  批次 {bi+1}/{len(batches)} ({len(chunk)} 条)...", end=" ", flush=True)
    raw = _call_llm(model, CLASSIFY_SYSTEM, prompt)
    results = _parse_json_array(raw)

    ok = 0
    for r in results:
      if isinstance(r, dict) and "id" in r:
        pi = r["id"] - 1
        if 0 <= pi < len(chunk):
          classes[indices[pi]] = r
          ok += 1
    print(f"完成 {ok}/{len(chunk)}")

    for idx, _ in chunk:
      if idx not in classes:
        classes[idx] = {"category": "classic_question", "situation": "any", "tags": []}

  return classes


def step_adapt(model, items: list[tuple[int, str]], batch_size: int = 10) -> dict[int, str]:
  """Step 3（可选）: 直播化改写"""
  print(f"\n{'='*50}")
  print(f"[Step 3] 直播化改写（{len(items)} 条，每批 {batch_size}）")
  print(f"{'='*50}")

  adapted: dict[int, str] = {}
  batches = _batch(items, batch_size)

  for bi, chunk in enumerate(batches):
    texts = [t for _, t in chunk]
    indices = [idx for idx, _ in chunk]
    prompt = ADAPT_USER_TEMPLATE.format(
      count=len(chunk),
      items=_format_numbered(texts),
    )

    print(f"  批次 {bi+1}/{len(batches)} ({len(chunk)} 条)...", end=" ", flush=True)
    raw = _call_llm(model, ADAPT_SYSTEM, prompt)
    results = _parse_json_array(raw)

    ok = 0
    for r in results:
      if isinstance(r, dict) and "id" in r and "text" in r:
        pi = r["id"] - 1
        if 0 <= pi < len(chunk):
          adapted[indices[pi]] = r["text"]
          ok += 1
    print(f"完成 {ok}/{len(chunk)}")

  return adapted


# ============================================================
# 输出
# ============================================================

def load_existing_corpus(path: Path) -> tuple[list[dict], set[str]]:
  entries: list[dict] = []
  texts: set[str] = set()
  if path.exists():
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
      line = line.strip()
      if not line:
        continue
      try:
        entry = json.loads(line)
        entries.append(entry)
        texts.add(entry.get("text", ""))
      except json.JSONDecodeError:
        pass
  return entries, texts


def write_corpus(entries: list[dict], path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    for entry in entries:
      f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ensure_meta(output_dir: Path):
  """在输出目录生成默认 meta.json（已存在则跳过）"""
  meta_path = output_dir / "meta.json"
  if meta_path.exists():
    return

  meta = {
    "description": "风格参考库",
    "retrieval_count": 3,
    "injection_header": (
      "【风格参考——吸收以下示例的语气、逻辑方式和表达节奏，"
      "不要直接照搬内容】"
    ),
    "categories": {
      "classic_question": "经典荒诞假设问题",
      "reasoning_chain": "完整推理链示范",
      "comment_reaction": "对弹幕的反应方式",
      "scene_reaction": "对画面的反应方式",
      "ice_breaker": "冷场时的主动发言",
      "comeback": "被质疑时的回击方式",
    },
  }
  meta_path.parent.mkdir(parents=True, exist_ok=True)
  meta_path.write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"已生成默认 meta.json → {meta_path}")


def print_stats(entries: list[dict]):
  cats: dict[str, int] = {}
  sits: dict[str, int] = {}
  scs: dict[int, int] = {}
  for e in entries:
    c = e.get("category", "?")
    s = e.get("situation", "?")
    sc = e.get("score", 0)
    cats[c] = cats.get(c, 0) + 1
    sits[s] = sits.get(s, 0) + 1
    scs[sc] = scs.get(sc, 0) + 1

  print(f"\n{'='*50}")
  print(f"语料统计: {len(entries)} 条")
  print(f"{'='*50}")
  print("\n分类分布:")
  for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    bar = "█" * v
    print(f"  {k:20s} {v:3d}  {bar}")
  print("\n场景分布:")
  for k, v in sorted(sits.items(), key=lambda x: -x[1]):
    bar = "█" * v
    print(f"  {k:20s} {v:3d}  {bar}")
  print("\n质量分布:")
  for sc in sorted(scs.keys(), reverse=True):
    bar = "█" * scs[sc]
    print(f"  {sc}分  {scs[sc]:3d}  {bar}")


# ============================================================
# 入口
# ============================================================

def main():
  parser = argparse.ArgumentParser(
    description="风格参考库语料处理管线",
    epilog=(
      "示例:\n"
      "  python tools/build_style_bank.py raw.txt "
      "-o personas/dacongming/style_bank/corpus.jsonl\n"
      "  python tools/build_style_bank.py raw.txt "
      "-o corpus.jsonl --model anthropic --min-score 4\n"
      "  python tools/build_style_bank.py more.txt "
      "-o corpus.jsonl --append"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("input", help="原始文本文件路径（UTF-8）")
  parser.add_argument("-o", "--output", required=True, help="输出 JSONL 路径")
  parser.add_argument(
    "--model", default="anthropic",
    choices=["openai", "anthropic", "gemini"],
    help="模型提供商（默认 anthropic）",
  )
  parser.add_argument(
    "--min-score", type=int, default=3,
    help="最低质量分，低于此分的条目被过滤（默认 3）",
  )
  parser.add_argument(
    "--adapt", action="store_true",
    help="启用直播化改写（将书面语转为口语）",
  )
  parser.add_argument(
    "--append", action="store_true",
    help="追加模式（保留已有条目，自动去重）",
  )
  parser.add_argument(
    "--id-prefix", default="rz",
    help="条目 ID 前缀（默认 rz）",
  )
  args = parser.parse_args()

  # ---- 读取输入 ----
  input_path = Path(args.input)
  if not input_path.exists():
    print(f"错误: 文件不存在 → {input_path}")
    sys.exit(1)

  raw = input_path.read_text(encoding="utf-8")
  items = parse_raw_text(raw)
  print(f"读取 {input_path.name}: 解析出 {len(items)} 条有效内容")

  if not items:
    print("没有有效内容，退出")
    sys.exit(0)

  # ---- 追加模式去重 ----
  output_path = Path(args.output)
  existing_entries: list[dict] = []
  existing_texts: set[str] = set()

  if args.append:
    existing_entries, existing_texts = load_existing_corpus(output_path)
    before = len(items)
    items = [x for x in items if x not in existing_texts]
    dup = before - len(items)
    if dup:
      print(f"追加模式: 跳过 {dup} 条已存在内容，剩余 {len(items)} 条")

  if not items:
    print("全部重复，无需处理")
    sys.exit(0)

  # ---- 初始化模型 ----
  provider = ModelType(args.model)
  print(f"\n模型: {provider.value} (small)")
  try:
    model = ModelProvider.remote_small(provider=provider)
  except ValueError as e:
    print(f"错误: {e}")
    sys.exit(1)

  # ---- Step 1: 评分 ----
  scores = step_score(model, items)

  passed: list[tuple[int, str, dict]] = []
  for idx, text in enumerate(items):
    info = scores.get(idx, {})
    if info.get("score", 0) >= args.min_score:
      passed.append((idx, text, info))

  dropped = len(items) - len(passed)
  print(f"\n过滤结果: 丢弃 {dropped} 条 (< {args.min_score}分)，保留 {len(passed)} 条")

  if not passed:
    print("没有通过筛选的内容")
    sys.exit(0)

  # ---- Step 2: 分类 ----
  to_cls = [(idx, text) for idx, text, _ in passed]
  classifications = step_classify(model, to_cls)

  # ---- Step 3: 改写（可选）----
  adapted: dict[int, str] = {}
  if args.adapt:
    to_adapt = [(idx, text) for idx, text, _ in passed]
    adapted = step_adapt(model, to_adapt)
  else:
    print(f"\n{'='*50}")
    print("[Step 3] 直播化改写（已跳过，用 --adapt 启用）")
    print(f"{'='*50}")

  # ---- 组装输出 ----
  start_num = len(existing_entries) + 1 if args.append else 1
  new_entries: list[dict] = []

  for i, (idx, text, score_info) in enumerate(passed):
    cls = classifications.get(idx, {})
    final_text = adapted.get(idx, text)
    new_entries.append({
      "id": f"{args.id_prefix}_{start_num + i:03d}",
      "text": final_text,
      "category": cls.get("category", "classic_question"),
      "situation": cls.get("situation", "any"),
      "tags": cls.get("tags", []),
      "score": score_info.get("score", 3),
    })

  all_entries = existing_entries + new_entries if args.append else new_entries
  write_corpus(all_entries, output_path)

  new_label = f"{len(new_entries)} 条新增" if args.append else f"{len(new_entries)} 条"
  total_label = f"共 {len(all_entries)} 条" if args.append else ""
  print(f"\n已写入 → {output_path}  ({new_label}{', ' + total_label if total_label else ''})")

  # ---- meta.json ----
  if output_path.name == "corpus.jsonl":
    ensure_meta(output_path.parent)

  # ---- 统计 ----
  print_stats(all_entries)


if __name__ == "__main__":
  main()
