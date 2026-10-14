"""CLI: python -m app assets/images --classifier both"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser(description="Detecta, segmenta y clasifica avisos publicitarios.")
    ap.add_argument("target", nargs="?", default="assets/images", help="Imagen o carpeta (por defecto: assets/images)")
    ap.add_argument("--out", default="output", help="Carpeta de salida")
    ap.add_argument("--segmenter", choices=["sam3", "dino-sam2"], default="sam3")
    ap.add_argument("--classifier", choices=["haiku", "openai", "both", "none"], default="haiku",
                    help="'both' clasifica con los dos; la categoría final es la de haiku")
    ap.add_argument("--prompts", help="Frases del detector separadas por coma (ej: 'sign,billboard')")
    ap.add_argument("--box-threshold", type=float, help="Umbral del detector (menor = más avisos y más falsos positivos)")
    ap.add_argument("--workers", type=int, default=8, help="Llamadas a la API en paralelo")
    ap.add_argument("--min-size", type=int, default=60, help="Descarta objetos cuyo lado mayor mida menos (px)")
    ap.add_argument("--min-score", type=float, default=0.5, help="Descarta detecciones con score menor")
    ap.add_argument("--skip-done", action="store_true", help="Omite imágenes que ya tienen resultado.json")
    args = ap.parse_args()

    from .classify import build_classifiers
    from .pipeline import append_csv, iter_images, process_image
    from .segment import build_segmenter

    try:
        classifiers = build_classifiers(args.classifier)
        print(f"Cargando segmentador {args.segmenter}…", flush=True)
        segmenter = build_segmenter(args.segmenter)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    target, out = Path(args.target), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = [p.strip() for p in args.prompts.split(",")] if args.prompts else None
    images = list(iter_images(target))
    if not images:
        print(f"No hay imágenes en {target}", file=sys.stderr)
        return 1

    csv_path = out / "resultados.csv"
    backends = [c.name for c in classifiers]
    total = Counter()
    for n, path in enumerate(images, 1):
        if args.skip_done and (out / path.stem / "resultado.json").exists():
            continue
        t = time.time()
        res = process_image(path, segmenter, classifiers, out, prompts, workers=args.workers, min_size=args.min_size, min_score=args.min_score, box_threshold=args.box_threshold)
        append_csv(csv_path, res, backends)
        c = Counter(a["categoria"] for a in res["avisos"])
        total.update(c)
        resumen = ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) or "sin avisos"
        print(f"[{n}/{len(images)}] {path.name}: {resumen} (descartados: {len(res['descartados'])}) [{time.time() - t:.0f}s]", flush=True)
    print(f"\nTotal: {dict(total)}  →  {out}/ (resumen en {csv_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
