"""Preparar la copia local de Rugby model v3 para las tres clases del proyecto.

Requiere el ZIP original extraído en data/datasets/rugby_tokyo_v3/original.
No descarga datos ni inicia entrenamiento. Conserva los archivos originales.
"""

import json
import math
from pathlib import Path
import shutil
from collections import Counter

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "datasets" / "rugby_tokyo_v3"


def main():
    source = BASE / "original"
    target = BASE / "prepared"
    metadata = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    names = metadata["names"]
    wanted = ["player", "ball", "referee"]
    mapping = {names.index(name): i for i, name in enumerate(wanted)}
    target.mkdir(exist_ok=False)
    report = {}
    for src_split, split in (("train", "train"), ("valid", "val"), ("test", "test")):
        image_dir = target / "images" / split
        label_dir = target / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        counts = Counter()
        n_images = n_empty = discarded = 0
        for image in sorted((source / src_split / "images").iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            with Image.open(image) as opened:
                opened.verify()
            label = source / src_split / "labels" / image.with_suffix(".txt").name
            lines = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                if (len(parts) != 5 or cls not in range(len(names))
                    or not all(math.isfinite(v) and 0 <= v <= 1 for v in coords)
                    or coords[2] <= 0 or coords[3] <= 0):
                    raise ValueError(f"Etiqueta inválida: {label}: {line}")
                if cls in mapping:
                    lines.append(f"{mapping[cls]} {' '.join(parts[1:])}")
                    counts[wanted[mapping[cls]]] += 1
                else:
                    discarded += 1
            # Una imagen etiquetada solo con eventos no demuestra ausencia de
            # jugadores/pelota/árbitros. No convertirla en un negativo falso.
            if not lines:
                n_empty += 1
                continue
            shutil.copy2(image, image_dir / image.name)
            (label_dir / label.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n_images += 1
        report[split] = {"images": n_images, "excluded_images_without_target_boxes": n_empty,
                         "boxes": dict(counts), "excluded_event_boxes": discarded}
    (BASE / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (target / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
        "names": dict(enumerate(wanted))}, sort_keys=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
