"""Evaluar best.pt en las imágenes de prueba reservadas, sin entrenar.

Usar el botón de correr con .venv-gpu. --check verifica sin medir.
Los resultados describen este dataset, no garantizan desempeño en otro partido.
"""
import argparse
import csv
from datetime import datetime
import json
from multiprocessing import freeze_support
from pathlib import Path

from yolo_runtime import ROOT, RUN, setup, local_dataset, check_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=RUN / "weights/best.pt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("batch debe ser positivo e imgsz múltiplo de 32.")
    setup()
    from ultralytics import YOLO
    import ultralytics
    import torch
    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"No existen los pesos: {weights}")
    check_gpu(args.device)
    data = local_dataset()
    model = YOLO(str(weights))
    if model.names != {0: "player", 1: "ball", 2: "referee"}:
        raise SystemExit(f"Las clases de este modelo no coinciden: {model.names}")
    if args.check:
        print(f"Evaluación preparada para {weights}. No se ejecutó la evaluación.")
        return
    import yaml
    dataset_config = yaml.safe_load(data.read_text(encoding="utf-8"))
    image_count = len(list((Path(dataset_config["path"]) / dataset_config["test"]).glob("*.jpg")))
    out = ROOT / "outputs/evaluation" / datetime.now().strftime("yolo26n_test_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    metrics = model.val(data=str(data), split="test", device=args.device, batch=args.batch,
        imgsz=args.imgsz, workers=0, plots=True, conf=0.001, iou=0.7,
        project=str(out.parent), name=out.name, save_dir=str(out), exist_ok=True, verbose=True)
    per_class = metrics.summary()
    report = {"weights": str(weights), "dataset": str(data), "split": "test",
        "images": image_count, "imgsz": args.imgsz, "batch": args.batch,
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
        "speed_ms_per_image": metrics.speed, "per_class": per_class,
        "note": "Test reservado de Tokyo; independencia por partido no certificada. Tiempos de val, no FPS de video completo."}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if per_class:
        with (out / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)
    text = [f"Modelo: {weights}", f"Evaluación: test reservado ({image_count} imágenes)",
        f"Precisión: {metrics.box.mp:.1%}", f"Recall: {metrics.box.mr:.1%}",
        f"mAP50: {metrics.box.map50:.1%}", f"mAP50-95: {metrics.box.map:.1%}",
        f"Inferencia: {metrics.speed.get('inference', 0):.2f} ms/imagen",
        "", "Por clase:"]
    text += [f"{row['Class']}: precisión {row['Box-P']:.1%}, recall {row['Box-R']:.1%}, mAP50 {row['mAP50']:.1%}, mAP50-95 {row['mAP50-95']:.1%}" for row in per_class]
    text += ["", report["note"]]
    (out / "resumen.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"Informe, CSV y gráficos: {out}")


if __name__ == "__main__":
    freeze_support()
    main()
