"""Clasifica solo los objetos con etiqueta manual y mide la precisión. Sirve para comparar
modelos o criterios sin reclasificar el lote completo.

    python -m app.eval_subset output etiquetas/etiquetas_manuales.csv --model claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from .classify import HaikuClassifier

Image.MAX_IMAGE_PIXELS = None


def _report(name: str, pairs: list[tuple[str, str]]) -> None:
    ok = sum(e == p for e, p in pairs)
    vf = [(e, p) for e, p in pairs if p in ("Valla", "Fachada")]
    real = [(e, p) for e, p in pairs if e in ("Valla", "Fachada")]
    print(
        f"  {name:16} exacta {ok:3}/{len(pairs)} ({100 * ok / len(pairs):.0f}%)"
        f" · V/F dibujadas correctas {sum(e == p for e, p in vf)}/{len(vf)}"
        f" · V/F reales encontradas {sum(e == p for e, p in real)}/{len(real)}"
    )


def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Carpeta con resultado.json por imagen (detecciones)")
    ap.add_argument("labels_csv")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="JSON con las predicciones (por defecto etiquetas/pred_<modelo>.json)")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.labels_csv, encoding="utf-8")))
    clf = HaikuClassifier(model=a.model, effort=a.effort)
    by_image: dict[str, list] = {}
    for r in rows:
        res = json.loads((Path(a.source) / r["imagen"] / "resultado.json").read_text())
        obj = next(x for x in res["avisos"] + res["descartados"] if x["id"] == int(r["id"]))
        by_image.setdefault(res["imagen"], []).append((r, obj))

    preds: dict[str, dict] = {}
    for img_path, items in by_image.items():
        image = Image.open(img_path).convert("RGB")

        def run(item):
            r, obj = item
            try:
                return r, clf.classify(image, obj["bbox_xyxy"]).to_dict()
            except Exception as e:  # un error de API no debe tumbar la evaluación
                return r, {"label": "Dudoso", "razon": f"error: {type(e).__name__}: {e}"[:300]}

        with ThreadPoolExecutor(a.workers) as pool:
            for r, c in pool.map(run, items):
                preds[f"{r['imagen']}#{r['id']}"] = c
        print(f"{img_path.split('/')[-1]}: {len(items)}", flush=True)

    out = Path(a.out or f"etiquetas/pred_{a.model}.json")
    out.write_text(json.dumps(preds, ensure_ascii=False, indent=1))
    errors = sum(c.get("razon", "").startswith("error") for c in preds.values())

    pairs = [(r["etiqueta"], preds[f"{r['imagen']}#{r['id']}"]["label"], r) for r in rows]
    print(f"\n{a.model} — errores de API: {errors} — predicciones en {out}")
    alea = [p for p in pairs if p[2]["grupo"] == "aleatorio"]
    # Mitad de la muestra aleatoria que no se usó para decidir criterios ni ejemplos.
    held = [p for p in alea if random.Random(f"{p[2]['imagen']}#{p[2]['id']}").random() < 0.5]
    for name, sub in [("aleatorio", alea), ("aleatorio-mitad", held), ("difíciles 80", [p for p in pairs if p[2]["grupo"] != "aleatorio"])]:
        if sub:
            _report(name, [(e, p) for e, p, _ in sub])
    print("\nErrores (aleatorio, real → predicho):")
    for (e, p), n in Counter((e, p) for e, p, _ in alea if e != p).most_common():
        print(f"  {e:13} → {p:13} {n}")


if __name__ == "__main__":
    main()
