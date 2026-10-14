"""Traslada etiquetas manuales de una carpeta a otra comparando cajas, no números de objeto.

    python -m app.remap_labels output_v4 output_v5 etiquetas/etiquetas_manuales.csv etiquetas/etiquetas_v5.csv

La numeración cambia entre reclasificaciones (filtro de calidad, unión de fragmentos de
valla), así que cada etiqueta se busca por su recuadro: misma caja, o la caja nueva que
contiene al menos el 80 % de la vieja (un fragmento que se unió a su valla).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

# Si varios fragmentos etiquetados caen en el mismo objeto nuevo, gana la etiqueta más "positiva".
PRIORITY = ["Valla", "Fachada", "Dudoso", "NoPublicidad"]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contained(old, new) -> float:
    inter = _area([max(old[0], new[0]), max(old[1], new[1]), min(old[2], new[2]), min(old[3], new[3])])
    return inter / max(1.0, _area(old))


def _objects(folder: Path, stem: str) -> list[dict]:
    res = json.loads((folder / stem / "resultado.json").read_text())
    return res["avisos"] + res["descartados"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_dir")
    ap.add_argument("new_dir")
    ap.add_argument("labels_csv")
    ap.add_argument("out_csv")
    a = ap.parse_args()
    old_dir, new_dir = Path(a.old_dir), Path(a.new_dir)

    rows = list(csv.DictReader(open(a.labels_csv, encoding="utf-8")))
    target: dict[tuple[str, int], list[dict]] = defaultdict(list)
    lost = []
    for r in rows:
        old = next(o for o in _objects(old_dir, r["imagen"]) if o["id"] == int(r["id"]))["bbox_xyxy"]
        cands = [(_contained(old, n["bbox_xyxy"]), -_area(n["bbox_xyxy"]), n["id"]) for n in _objects(new_dir, r["imagen"])]
        best = max(cands)
        if best[0] < 0.8:
            lost.append(r)
            continue
        target[(r["imagen"], best[2])].append(r)

    out = []
    for (stem, new_id), rs in target.items():
        labels = sorted({r["etiqueta"] for r in rs}, key=PRIORITY.index)
        if len(rs) > 1:
            print(f"  {stem}: {', '.join('#' + r['id'] for r in rs)} → #{new_id} ({' / '.join(labels)} → {labels[0]})")
        out.append({"imagen": stem, "id": new_id, "grupo": rs[0]["grupo"], "etiqueta": labels[0]})

    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["imagen", "id", "grupo", "etiqueta"])
        w.writeheader()
        w.writerows(out)
    moved = sum(1 for (stem, nid), rs in target.items() for r in rs if int(r["id"]) != nid)
    print(f"{len(rows)} etiquetas → {len(out)} objetos ({moved} cambiaron de número, {len(lost)} sin pareja) → {a.out_csv}")
    for r in lost:
        print(f"  sin pareja: {r['imagen']} #{r['id']} {r['etiqueta']}")


if __name__ == "__main__":
    main()
