"""Precisión de cada clasificador contra las etiquetas manuales.

    python -m app.evaluate output_v2 ~/Downloads/etiquetas_manuales.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["Valla", "Fachada", "Dudoso", "NoPublicidad"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("labels_csv")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)

    preds: dict[tuple[str, int], dict[str, str]] = {}
    for rj in out_dir.glob("*/resultado.json"):
        res = json.loads(rj.read_text())
        for r in res["avisos"] + res["descartados"]:
            p = {b: c["label"] for b, c in r["clasificacion"].items()}
            p["final"] = r["categoria"]
            preds[(rj.parent.name, r["id"])] = p

    rows = list(csv.DictReader(open(Path(a.labels_csv).expanduser(), encoding="utf-8")))
    rows = [r for r in rows if (r["imagen"], int(r["id"])) in preds]
    if not rows:
        raise SystemExit("El CSV no coincide con ningún objeto de esta carpeta.")
    systems = list(preds[(rows[0]["imagen"], int(rows[0]["id"]))].keys())

    for grupo in ["todos", "desacuerdo", "control"]:
        sub = rows if grupo == "todos" else [r for r in rows if r["grupo"] == grupo]
        if not sub:
            continue
        print(f"\n== {grupo} ({len(sub)} objetos) ==")
        for s in systems:
            ok = sum(preds[(r["imagen"], int(r["id"]))][s] == r["etiqueta"] for r in sub)
            print(f"  {s:8} acierta {ok:3}/{len(sub)}  ({100 * ok / len(sub):.0f}%)")

    print("\nAciertos por categoría real (todos):")
    print(f"  {'real':13}" + "".join(f"{s:>10}" for s in systems))
    for lab in LABELS:
        sub = [r for r in rows if r["etiqueta"] == lab]
        if sub:
            cells = "".join(
                f"{sum(preds[(r['imagen'], int(r['id']))][s] == lab for r in sub):>5}/{len(sub):<4}" for s in systems
            )
            print(f"  {lab:13}{cells}")

    print("\nErrores más comunes de 'final' (real → predicho):")
    errs = Counter((r["etiqueta"], preds[(r["imagen"], int(r["id"]))]["final"]) for r in rows)
    for (real, pred), n in errs.most_common():
        if real != pred:
            print(f"  {real:13} → {pred:13} {n}")


if __name__ == "__main__":
    main()
