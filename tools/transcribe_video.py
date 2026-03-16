"""
用 Whisper 转录视频音频，输出带时间戳的文字稿。

用法:
  python tools/transcribe_video.py data/example_videos/neuro_love_death_robots.wav
"""

import json
import sys
from pathlib import Path

import whisper


def format_time(seconds: float) -> str:
  m, s = divmod(seconds, 60)
  return f"{int(m):02d}:{s:05.2f}"


def main():
  audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/example_videos/neuro_love_death_robots.wav"
  audio_path = Path(audio_path)
  stem = audio_path.stem
  out_dir = audio_path.parent

  print(f"Loading Whisper medium model...")
  model = whisper.load_model("medium")

  print(f"Transcribing: {audio_path}")
  result = model.transcribe(
    str(audio_path),
    language="en",
    verbose=True,
  )

  segments = []
  for seg in result["segments"]:
    segments.append({
      "start": round(seg["start"], 2),
      "end": round(seg["end"], 2),
      "text": seg["text"].strip(),
    })

  json_path = out_dir / f"{stem}_transcript.json"
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(segments, f, ensure_ascii=False, indent=2)

  txt_path = out_dir / f"{stem}_transcript.txt"
  with open(txt_path, "w", encoding="utf-8") as f:
    for seg in segments:
      ts = f"[{format_time(seg['start'])} -> {format_time(seg['end'])}]"
      f.write(f"{ts}  {seg['text']}\n")

  print(f"\nDone! {len(segments)} segments")
  print(f"  JSON: {json_path}")
  print(f"  TXT:  {txt_path}")


if __name__ == "__main__":
  main()
