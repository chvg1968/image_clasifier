"""Reclasifica detecciones ya guardadas, sin volver a correr la segmentación.

    python -m app.reclassify output output_v2 --classifier both --min-size 60 --min-score 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from .classify import build_classifiers
from .pipeline import append_csv, load_detections, split_by_quality, write_outputs
from .segment import merge_billboard_parts


def main() -> int:
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Carpeta con resultados previos (output/)")
    ap.add_argument("out", help="Carpeta nueva de salida")
    ap.add_argument("--classifier", choices=["haiku", "openai", "both", "none"], default="haiku")
    ap.add_argument("--min-size", type=int, default=60)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--only", help="Nombres de imagen separados por coma (ej: L6_070,L6_098)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    try:
        classifiers = build_classifiers(args.classifier)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    src, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsons = sorted(src.glob("*/resultado.json"))
    if args.only:
        wanted = set(args.only.split(","))
        jsons = [j for j in jsons if j.parent.name in wanted]

    total = Counter()
    for n, rj in enumerate(jsons, 1):
        t = time.time()
        path, dets = load_detections(rj)
        dets = merge_billboard_parts(dets)
        keep, drop = split_by_quality(dets, args.min_size, args.min_score)
        image = Image.open(path).convert("RGB")
        res = write_outputs(path, image, keep, classifiers, out, args.workers, filtered=drop, min_size=args.min_size, min_score=args.min_score)
        append_csv(out / "resultados.csv", res, [c.name for c in classifiers])
        c = Counter(a["categoria"] for a in res["avisos"])
        total.update(c)
        resumen = ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) or "sin avisos"
        print(f"[{n}/{len(jsons)}] {path.name}: {resumen} (descartados: {len(res['descartados'])}) [{time.time() - t:.0f}s]", flush=True)
    print(f"\nTotal: {dict(total)}  →  {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
