"""Run pretrained YOLO on every frame and export an annotated MP4 and CSVs."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import cv2
import imageio_ffmpeg
import torch
import ultralytics
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path("weights/yolo26n.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/yolo26n_10s"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--source-start", type=float, default=0)
    args = parser.parse_args()
    if not args.video.is_file() or not args.model.is_file():
        parser.error("Video and model must already exist locally")
    if args.imgsz <= 0 or not 0 < args.conf < 1:
        parser.error("imgsz must be positive and conf must be between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=False)

    cap = cv2.VideoCapture(str(args.video))
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or fps <= 0:
            raise RuntimeError("Could not decode video or determine FPS")
        height, width = frame.shape[:2]
        load_start = time.perf_counter()
        model = YOLO(str(args.model))
        load_seconds = time.perf_counter() - load_start
        options = dict(device="cpu", imgsz=args.imgsz, conf=args.conf, verbose=False)
        warmup_start = time.perf_counter()
        model.predict(frame, **options)
        warmup_seconds = time.perf_counter() - warmup_start

        writer = imageio_ffmpeg.write_frames(
            str(args.output / "annotated.mp4"), (width, height),
            fps=fps, pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            codec="libx264", quality=7, macro_block_size=1,
            output_params=["-preset", "veryfast", "-movflags", "+faststart"],
        )
        writer.send(None)
        counts = Counter()
        frames_with_class = Counter()
        inference_ms = 0.0
        count = 0
        started = time.perf_counter()
        with (args.output / "detections.csv").open("w", newline="", encoding="utf-8") as det_file, \
             (args.output / "frames.csv").open("w", newline="", encoding="utf-8") as frame_file:
            detections = csv.writer(det_file)
            detections.writerow(["frame", "clip_seconds", "source_seconds", "class_id",
                                 "class_name", "confidence", "x1", "y1", "x2", "y2"])
            frames = csv.writer(frame_file)
            frames.writerow(["frame", "clip_seconds", "detections", "inference_ms"])
            while ok:
                result = model.predict(frame, **options)[0]
                seconds = count / fps
                labels = set()
                for box, cls, conf in zip(result.boxes.xyxy.tolist(),
                                          result.boxes.cls.tolist(),
                                          result.boxes.conf.tolist()):
                    label = result.names[int(cls)]
                    counts[label] += 1
                    labels.add(label)
                    detections.writerow([count, f"{seconds:.6f}",
                                         f"{args.source_start + seconds:.6f}", int(cls),
                                         label, f"{conf:.6f}", *[round(x, 2) for x in box]])
                frames_with_class.update(labels)
                ms = float(result.speed["inference"])
                inference_ms += ms
                frames.writerow([count, f"{seconds:.6f}", len(result.boxes), round(ms, 3)])
                annotated = result.plot(line_width=2, font_size=12)
                writer.send(annotated)
                if count in (0, 90, 180, 270):
                    cv2.imwrite(str(args.output / f"preview_{count:04d}.jpg"), annotated)
                count += 1
                if count % 30 == 0:
                    print(f"Processed {count}/{expected_frames} frames", flush=True)
                ok, frame = cap.read()
        writer.close()
        writer = None
        elapsed = time.perf_counter() - started
        if count != expected_frames:
            raise RuntimeError(f"Decoded {count} frames; expected {expected_frames}")
        summary = {
            "video": str(args.video.resolve()), "model": str(args.model.resolve()),
            "ultralytics": ultralytics.__version__, "torch": torch.__version__,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "imgsz": args.imgsz, "confidence_threshold": args.conf,
            "source_start_seconds": args.source_start,
            "width": width, "height": height, "fps": fps, "frames": count,
            "clip_seconds": count / fps,
            "model_load_seconds": round(load_seconds, 3),
            "warmup_seconds": round(warmup_seconds, 3),
            "processing_seconds": round(elapsed, 3),
            "processing_fps": round(count / elapsed, 3),
            "mean_inference_ms": round(inference_ms / count, 3),
            "detections_by_class": dict(counts),
            "frames_with_class": dict(frames_with_class),
            "notes": [
                "Processing time includes decoding after the first frame, prediction, plotting, CSV and MP4 encoding.",
                "Imports, model loading and one warmup prediction are excluded from processing time.",
                "Detection counts are repeated observations across frames, not unique objects or players.",
                "No training, tracking, team assignment or accuracy evaluation was performed.",
            ],
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
