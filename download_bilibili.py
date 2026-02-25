"""
B站视频 + 弹幕下载工具
下载视频文件（mp4）和弹幕 XML，供 VLM Demo 使用

依赖:
  pip install yt-dlp requests

用法:
  python download_bilibili.py "https://b23.tv/ogv8dsh"
  python download_bilibili.py "https://www.bilibili.com/video/BV1xx..."
  python download_bilibili.py "https://www.bilibili.com/video/BV1xx..." --output-dir data/my_video
  python download_bilibili.py "BV1xxxxxxxxxx"
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path
from urllib.parse import urlparse

import requests

BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
DANMAKU_XML_API = "https://comment.bilibili.com/{cid}.xml"

HEADERS = {
  "User-Agent": (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
  ),
  "Referer": "https://www.bilibili.com/",
}


def resolve_short_url(url: str) -> str:
  """解析 b23.tv 短链接 → 完整 URL"""
  if "b23.tv" in url:
    print(f"  解析短链接: {url}")
    resp = requests.head(url, allow_redirects=True, headers=HEADERS, timeout=10)
    url = resp.url
    print(f"  → {url}")
  return url


def extract_bvid(url_or_bvid: str) -> str:
  """从 URL 或直接输入中提取 BV 号"""
  # 直接是 BV 号
  if re.match(r"^BV[0-9A-Za-z]+$", url_or_bvid):
    return url_or_bvid

  # 从 URL 中提取
  match = re.search(r"(BV[0-9A-Za-z]+)", url_or_bvid)
  if match:
    return match.group(1)

  raise ValueError(f"无法从输入中提取 BV 号: {url_or_bvid}")


def get_video_info(bvid: str) -> dict:
  """通过 B站 API 获取视频信息（含 cid）"""
  resp = requests.get(
    BILIBILI_VIEW_API,
    params={"bvid": bvid},
    headers=HEADERS,
    timeout=10,
  )
  data = resp.json()

  if data.get("code") != 0:
    raise RuntimeError(f"B站 API 错误: {data.get('message', '未知错误')}")

  return data["data"]


def download_danmaku_xml(cid: int, output_path: Path) -> Path:
  """下载弹幕 XML 文件"""
  url = DANMAKU_XML_API.format(cid=cid)
  print(f"  下载弹幕 XML: {url}")

  resp = requests.get(url, headers=HEADERS, timeout=30)

  # B站弹幕 API 返回的内容可能是 deflate 压缩的，统一用 bytes 处理后 decode UTF-8
  raw = resp.content
  try:
    xml_text = zlib.decompress(raw, -zlib.MAX_WBITS).decode("utf-8")
  except zlib.error:
    try:
      xml_text = zlib.decompress(raw).decode("utf-8")
    except zlib.error:
      xml_text = raw.decode("utf-8")

  output_path.write_text(xml_text, encoding="utf-8")
  return output_path


def download_video_ytdlp(url: str, output_dir: Path, filename: str) -> Path:
  """用 yt-dlp 下载视频"""
  output_template = str(output_dir / f"{filename}.%(ext)s")

  # 优先使用 yt-dlp 可执行文件；若 PATH 不可见则回退到 python -m yt_dlp
  ytdlp_cmd = [shutil.which("yt-dlp")] if shutil.which("yt-dlp") else [sys.executable, "-m", "yt_dlp"]

  cmd = ytdlp_cmd + [
    "--format", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
    "--merge-output-format", "mp4",
    "--output", output_template,
    "--no-playlist",
    "--referer", "https://www.bilibili.com/",
    "--user-agent", HEADERS["User-Agent"],
    url,
  ]

  print(f"  执行: {' '.join(cmd)}")
  print()

  result = subprocess.run(cmd, capture_output=False)

  if result.returncode != 0:
    raise RuntimeError("yt-dlp 下载失败")

  # 找到下载的文件（yt-dlp 可能会加格式ID后缀如 .f30080.mp4）
  candidates = []
  for f in output_dir.iterdir():
    if f.name.startswith(filename) and f.suffix in (".mp4", ".mkv", ".webm", ".flv"):
      candidates.append(f)

  if not candidates:
    raise FileNotFoundError(f"未找到下载的视频文件: {output_dir / filename}.*")

  # 优先精确匹配，否则取最大的（通常是视频而非音频）
  exact = [f for f in candidates if f.stem == filename]
  if exact:
    return exact[0]
  return max(candidates, key=lambda f: f.stat().st_size)


def count_danmaku(xml_path: Path) -> int:
  """粗略统计弹幕数量"""
  content = xml_path.read_text(encoding="utf-8")
  return content.count("<d p=")


def main():
  parser = argparse.ArgumentParser(
    description="下载 B站视频 + 弹幕 XML（供 VLM Demo 使用）",
  )
  parser.add_argument(
    "url",
    help="B站视频链接（支持 b23.tv 短链、完整链接、BV号）",
  )
  parser.add_argument(
    "--output-dir", "-o",
    default="data",
    help="输出目录（默认 data/）",
  )
  parser.add_argument(
    "--skip-video",
    action="store_true",
    help="跳过视频下载（只下载弹幕 XML）",
  )
  parser.add_argument(
    "--page", "-p",
    type=int, default=1,
    help="分P编号（默认第1P）",
  )
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print("=" * 60)
  print("  B站视频 + 弹幕下载工具")
  print("=" * 60)
  print()

  # 1. 解析 URL
  print("[1/4] 解析链接...")
  url = resolve_short_url(args.url)
  bvid = extract_bvid(url)
  print(f"  BV号: {bvid}")
  print()

  # 2. 获取视频信息
  print("[2/4] 获取视频信息...")
  info = get_video_info(bvid)

  title = info["title"]
  owner = info["owner"]["name"]
  duration = info["duration"]
  pages = info.get("pages", [])

  safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:80]

  print(f"  标题: {title}")
  print(f"  UP主: {owner}")
  print(f"  时长: {duration // 60}分{duration % 60}秒")
  print(f"  分P数: {len(pages)}")

  # 获取目标分P的 cid
  page_idx = args.page - 1
  if page_idx >= len(pages):
    print(f"  错误: 只有 {len(pages)} 个分P，但请求了第 {args.page} P")
    sys.exit(1)

  page_info = pages[page_idx]
  cid = page_info["cid"]
  page_title = page_info.get("part", "")

  if len(pages) > 1:
    print(f"  选择: P{args.page} - {page_title}")
  print(f"  CID: {cid}")
  print()

  # 3. 下载弹幕 XML
  print("[3/4] 下载弹幕 XML...")
  xml_path = output_dir / f"{safe_title}.xml"
  download_danmaku_xml(cid, xml_path)
  dm_count = count_danmaku(xml_path)
  print(f"  保存到: {xml_path}")
  print(f"  弹幕数量: {dm_count}")
  print()

  # 4. 下载视频
  if args.skip_video:
    print("[4/4] 跳过视频下载")
    video_path = None
  else:
    print("[4/4] 下载视频（yt-dlp）...")
    full_url = f"https://www.bilibili.com/video/{bvid}"
    if len(pages) > 1:
      full_url += f"?p={args.page}"
    video_path = download_video_ytdlp(full_url, output_dir, safe_title)
    print(f"  保存到: {video_path}")
  print()

  # 完成
  print("=" * 60)
  print("  下载完成！")
  print("=" * 60)
  print()
  if video_path:
    print(f"  视频: {video_path}")
  print(f"  弹幕: {xml_path} ({dm_count} 条)")
  print()
  print("  运行 VLM Demo:")
  if video_path:
    print(f"  python run_vlm_demo.py --video \"{video_path}\" --danmaku \"{xml_path}\"")
  else:
    print(f"  python run_vlm_demo.py --video <视频文件> --danmaku \"{xml_path}\"")
  print()


if __name__ == "__main__":
  main()
