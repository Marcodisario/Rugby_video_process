"""Create timestamped thumbnails and a contact sheet from a local video."""

import argparse
import io
import json
import math
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/inspection"))
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--source-start", type=float, default=0)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("Interval must be positive and finite")
    args.output.mkdir(parents=True, exist_ok=True)
    reader = imageio_ffmpeg.read_frames(str(args.video))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    duration = metadata["duration"]
    count = max(1, math.ceil(duration / args.interval))
    sheet = Image.new("RGB", (480 * 3, 296 * math.ceil(count / 3)), "#15202b")
    draw = ImageDraw.Draw(sheet)
    for index in range(count):
        seconds = index * args.interval
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
             "-ss", str(seconds), "-i", str(args.video), "-frames:v", "1",
             "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
            check=True, capture_output=True,
        )
        frame = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        frame.save(args.output / f"frame_{seconds:07.2f}.jpg", quality=95)
        frame.thumbnail((480, 270))
        x, y = (index % 3) * 480, (index // 3) * 296
        sheet.paste(frame, (x, y))
        source = int(args.source_start + seconds)
        stamp = f"{source // 3600:02}:{source // 60 % 60:02}:{source % 60:02}"
        draw.text((x + 8, y + 275), f"Video {stamp} | clip +{seconds:g}s", fill="white")
    sheet.save(args.output / "contact_sheet.jpg", quality=92)
    (args.output / "metadata.json").write_text(
        json.dumps({"video": str(args.video), "source_start": args.source_start,
                    **metadata}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(args.output / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
