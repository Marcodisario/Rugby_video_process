"""Continuar el entrenamiento interrumpido. Ejecutar con el Python de .venv-gpu.

El botón de correr retoma last.pt; --check solo prepara y verifica sin entrenar.
"""
import argparse
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
import shutil

from yolo_runtime import RUN, setup, local_dataset, check_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=RUN / "weights/last.pt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    setup()
    import torch
    from ultralytics import YOLO

    checkpoint = args.weights.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"No existe el checkpoint: {checkpoint}")
    # Solo cargar checkpoints locales de confianza, generados por este proyecto.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", -1)
    total = ckpt.get("train_args", {}).get("epochs", 50)
    if epoch < 0 or ckpt.get("optimizer") is None or epoch + 1 >= total:
        raise SystemExit("El checkpoint no permite reanudar o ya completó las épocas previstas.")
    print(f"Checkpoint: {checkpoint}\nÉpocas completadas: {epoch + 1}/{total}", flush=True)
    del ckpt
    check_gpu(args.device)
    data = local_dataset()
    if args.check:
        print("Listo para reanudar. No se inició entrenamiento.")
        return
    run = checkpoint.parent.parent
    backup = run / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True, exist_ok=False)
    for file in (checkpoint, run / "weights/best.pt", run / "results.csv", run / "args.yaml"):
        if file.is_file():
            shutil.copy2(file, backup / file.name)
    model = YOLO(str(checkpoint))
    model.train(resume=True, data=str(data), device=args.device, workers=0, cache=False,
                save_dir=str(run))
    print(f"Entrenamiento terminado. Resultados en {run}")


if __name__ == "__main__":
    freeze_support()
    main()
