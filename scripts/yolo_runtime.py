"""Datos locales verificados fuera de OneDrive y configuración común de YOLO."""
import os
import json
import hashlib
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/training/yolo26n_rugby_20261105_232157"


def setup():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def local_dataset() -> Path:
    """Reconstruir imágenes desde el ZIP, con etiquetas adaptadas, y verificar lectura.

    No usa las imágenes sincronizadas durante entrenamiento. La copia local se
    compara con el ZIP en cada preparación, sin modificar los originales.
    """
    import cv2
    import numpy as np
    import yaml

    base = ROOT / "data/datasets/rugby_tokyo_v3"
    dest = Path.home() / "AppData/Local/RugbyVideoProcess/datasets/rugby_tokyo_v3"
    config = dest / "dataset.yaml"
    manifest = dest / "verified_files.json"
    if manifest.is_file() and config.is_file():
        hashes = json.loads(manifest.read_text(encoding="utf-8"))
        if all((dest / name).is_file() and hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items()):
            print(f"Dataset local verificado: {dest}", flush=True)
            return config
        print("La copia local necesita reparación desde el ZIP.", flush=True)
    counts, hashes = {}, {}
    with ZipFile(base / "original_yolo26.zip") as archive:
        for split, original in (("train", "train"), ("val", "valid"), ("test", "test")):
            labels = sorted(name for name in archive.namelist()
                            if name.startswith(f"{original}/labels/") and name.endswith(".txt"))
            if not labels:
                raise ValueError(f"No hay etiquetas en {split}")
            image_dir, label_dir = dest / "images" / split, dest / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            expected = set()
            for label in labels:
                # Misma adaptación que prepare_tokyo_dataset.py, sin depender
                # de archivos individuales sincronizados con OneDrive.
                lines = []
                for line in archive.read(label).decode("utf-8").splitlines():
                    parts = line.split()
                    cls = int(parts[0])
                    mapping = {3: 0, 0: 1, 4: 2}
                    if cls in mapping:
                        lines.append(f"{mapping[cls]} {' '.join(parts[1:])}")
                if not lines:
                    continue
                label_name = Path(label).name
                name = Path(label_name).stem + ".jpg"
                expected.add(name)
                content = archive.read(f"{original}/images/{name}")
                image = image_dir / name
                if not image.exists() or image.read_bytes() != content:
                    image.write_bytes(content)
                decoded = cv2.imdecode(np.frombuffer(image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f"No se puede decodificar {image}")
                local_label = label_dir / label_name
                local_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                for file in (image, local_label):
                    hashes[file.relative_to(dest).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
            actual = {p.name for p in image_dir.iterdir() if p.is_file()}
            if actual != expected:
                raise ValueError(f"Archivos inesperados en {image_dir}; no se borraron automáticamente.")
            counts[split] = len(expected)
    config.write_text(yaml.safe_dump({"path": str(dest), "train": "images/train",
        "val": "images/val", "test": "images/test",
        "names": {0: "player", 1: "ball", 2: "referee"}}, sort_keys=False), encoding="utf-8")
    hashes["dataset.yaml"] = hashlib.sha256(config.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(f"Dataset local verificado: {dest}\nImágenes: {counts}", flush=True)
    return config


def check_gpu(device=0):
    import torch
    if device < 0 or not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise ValueError("Se necesita una GPU NVIDIA disponible con PyTorch CUDA.")
    (torch.ones(1, device=f"cuda:{device}") * 2).sum().item()
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)


if __name__ == "__main__":
    setup()
    local_dataset()
