"""Download a bounded, video-only clip for computer vision experiments."""

import argparse
import math
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg


def timestamp(value: str) -> float:
    parts = value.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use seconds, MM:SS or HH:MM:SS") from exc
    if not 1 <= len(numbers) <= 3 or any(
        not math.isfinite(number) or number < 0 for number in numbers
    ):
        raise argparse.ArgumentTypeError("Timestamp must be finite and nonnegative")
    if any(number >= 60 for number in numbers[1:]):
        raise argparse.ArgumentTypeError("Minutes/seconds after ':' must be below 60")
    return sum(number * 60**index for index, number in enumerate(reversed(numbers)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--start", type=timestamp, required=True)
    parser.add_argument("--duration", type=timestamp, default=60)
    parser.add_argument("--height", type=int, choices=[360, 480, 720, 1080], default=1080)
    parser.add_argument("--output", type=Path, default=Path("data/clips"))
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("Duration must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    end = args.start + args.duration
    template = args.output / (
        f"%(id)s_{args.start:g}-{end:g}s_{args.height}p.%(ext)s"
    )
    subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--ignore-config", "--no-playlist", "--no-overwrites",
            "--js-runtimes", "node",
            "--ffmpeg-location", imageio_ffmpeg.get_ffmpeg_exe(),
            "-f", f"bv[ext=mp4][vcodec^=avc1][height<={args.height}]",
            "--download-sections", f"*{args.start:g}-{end:g}",
            "--force-keyframes-at-cuts",
            "-o", str(template), args.url,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
