"""Entrenamiento inicial de rugby con YOLO26n preentrenado, solo en GPU NVIDIA.

Ejecutar en la PC con GPU, una vez instalado PyTorch CUDA y etiquetado el dataset.
Ver docs/entrenamiento_gpu.md. Importar este archivo no inicia entrenamiento.
"""

import argparse
from datetime import datetime
from multiprocessing import freeze_support
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Configuración pequeña de inicio. También se puede ajustar mediante argumentos.
DEFAULTS = {
    "data": ROOT / "configs" / "rugby_dataset.yaml",
    "device": 0,       # Primera GPU NVIDIA visible. No se permite CPU.
    "epochs": 50,      # Máximo; early stopping puede terminar antes.
    "batch": 4,        # Bajar a 2 o 1 si falta memoria de GPU.
    "imgsz": 640,
    "workers": 0,      # Inicio sencillo y compatible con Windows.
}


def prepare_dataset(config: Path) -> dict:
    """Resolver rutas respecto del YAML y detectar un dataset vacío antes de entrenar."""
    import yaml

    if not config.is_file():
        raise ValueError(f"No existe el YAML del dataset: {config}")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("names"):
        raise ValueError("El YAML necesita names, train y val. Ver configs/rugby_dataset.yaml.")
    root = Path(data.get("path", "."))
    root = (config.parent / root).resolve()
    resolved = {"path": str(root), "names": data["names"]}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split in ("train", "val"):
        value = data.get(split)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{split} debe indicar una carpeta de imágenes.")
        images = (root / value).resolve()
        files = [p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions]
        if not files:
            raise ValueError(f"No hay imágenes para {split} en {images}. Primero preparar el dataset.")
        if "images" not in images.parts:
            raise ValueError(f"La ruta de {split} debe contener una carpeta llamada images.")
        index = len(images.parts) - 1 - images.parts[::-1].index("images")
        labels = Path(*images.parts[:index], "labels", *images.parts[index + 1:])
        # Exigir un TXT por imagen, incluso vacío para un fondo sin objetos.
        # Evita tratar imágenes sin etiquetar como ejemplos negativos por accidente.
        nonempty = 0
        for image in files:
            label = labels / image.relative_to(images).with_suffix(".txt")
            if not label.is_file():
                raise ValueError(f"Falta la etiqueta: {label}. Ver docs/entrenamiento_gpu.md.")
            nonempty += bool(label.read_text(encoding="utf-8").strip())
        if not nonempty:
            raise ValueError(f"Todas las etiquetas de {split} están vacías.")
        resolved[split] = str(images)
        print(f"{split}: {len(files)} imágenes, {nonempty} etiquetas no vacías.", flush=True)
    if resolved["train"] == resolved["val"]:
        raise ValueError("train y val deben ser carpetas distintas, con secuencias independientes.")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULTS["data"])
    for name in ("device", "epochs", "batch", "imgsz", "workers"):
        parser.add_argument(f"--{name}", type=int, default=DEFAULTS[name])
    args = parser.parse_args()
    if args.device < 0 or args.workers < 0 or min(args.epochs, args.batch, args.imgsz) <= 0:
        parser.error("GPU/workers deben ser >= 0; epochs, batch e imgsz deben ser > 0.")
    if args.imgsz % 32:
        parser.error("Usar imgsz múltiplo de 32, por ejemplo 640 o 960.")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import torch
    import yaml
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA no está disponible. Este script requiere GPU NVIDIA y PyTorch con CUDA. "
            "Revisar docs/entrenamiento_gpu.md; no se iniciará entrenamiento en CPU."
        )
    if args.device >= torch.cuda.device_count():
        raise SystemExit(f"No existe la GPU {args.device}. Disponibles: {torch.cuda.device_count()}.")
    # Verifica una operación real: is_available() solo no garantiza compatibilidad.
    try:
        probe = torch.ones(1, device=f"cuda:{args.device}")
        (probe * 2).sum().item()
        del probe
    except RuntimeError as exc:
        raise SystemExit(f"La instalación CUDA no puede ejecutar en esta GPU: {exc}") from exc
    print(f"GPU {args.device}: {torch.cuda.get_device_name(args.device)}", flush=True)
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)

    try:
        dataset = prepare_dataset(args.data.resolve())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    # El YAML resuelto queda junto a los resultados. No modifica el original.
    run_dir = ROOT / "outputs" / "training" / f"yolo26n_rugby_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_yaml = run_dir / "dataset.yaml"
    resolved_yaml.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    weights = ROOT / "weights" / "yolo26n.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    # Carga los pesos locales o descarga los oficiales si no están en la otra PC.
    model = YOLO(str(weights))
    try:
        model.train(
            data=str(resolved_yaml),
            device=args.device,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            pretrained=True,
            amp=True,                  # Precisión mixta, sujeta al chequeo de Ultralytics.
            patience=10,
            optimizer="auto",
            seed=42,
            deterministic=True,
            cache=False,               # No cargar todo el dataset en RAM.
            close_mosaic=min(10, args.epochs),
            val=True,
            plots=True,
            save=True,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,              # La carpeta se acaba de crear de forma exclusiva.
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("Memoria GPU insuficiente. Reintentar con --batch 2 o --batch 1.") from exc
    print(f"Mejores pesos: {run_dir / 'weights' / 'best.pt'}")
    print(f"Último checkpoint: {run_dir / 'weights' / 'last.pt'}")
    print(f"Métricas y gráficos: {run_dir}")


if __name__ == "__main__":
    freeze_support()  # Necesario para procesos de carga de datos en Windows.
    main()
