"""Aplica etiquetas manuales sobre los resultados y genera la carpeta de entrega.

    python -m app.apply_labels output_v2 etiquetas_manuales.csv entrega

En cada objeto etiquetado a mano, la etiqueta humana reemplaza la del modelo
(queda marcado con verificado_manual y conserva categoria_modelo). Se
regeneran anotada.jpg, máscaras, resultado.json y resultados.csv. No llama a
ninguna API ni vuelve a segmentar.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from .pipeline import CSV_FIELDS, annotate, append_csv, load_detections

Image.MAX_IMAGE_PIXELS = None
VALID = {"Valla", "Fachada", "Dudoso", "NoPublicidad"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("labels_csv")
    ap.add_argument("out")
    a = ap.parse_args()
    src, out = Path(a.source), Path(a.out)

    labels = {}
    for r in csv.DictReader(open(Path(a.labels_csv).expanduser(), encoding="utf-8")):
        if r["etiqueta"] not in VALID:
            raise SystemExit(f"Etiqueta inválida: {r}")
        labels[(r["imagen"], int(r["id"]))] = r["etiqueta"]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    applied, changed = 0, Counter()
    backends: list[str] = []

    for rj in sorted(src.glob("*/resultado.json")):
        stem = rj.parent.name
        res = json.loads(rj.read_text())
        dst = out / stem
        (dst / "masks").mkdir(parents=True)
        shutil.copytree(rj.parent / "crops", dst / "crops")

        records = sorted(res["avisos"] + res["descartados"], key=lambda r: r["id"])
        for r in records:
            backends = backends or list(r["clasificacion"].keys())
            r["categoria_modelo"] = r["categoria"]
            r["verificado_manual"] = False
            human = labels.get((stem, r["id"]))
            if human:
                applied += 1
                r["verificado_manual"] = True
                if human != r["categoria"]:
                    changed[(r["categoria"], human)] += 1
                r["categoria"] = human
            if r["mascara"]:
                new_mask = f"masks/{r['id']:02d}_{r['categoria']}.png"
                shutil.copy(rj.parent / r["mascara"], dst / new_mask)
                r["mascara"] = new_mask

        res["avisos"] = [r for r in records if r["categoria"] != "NoPublicidad"]
        res["descartados"] = [r for r in records if r["categoria"] == "NoPublicidad"]
        (dst / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))

        _, dets = load_detections(rj)
        image = Image.open(res["imagen"]).convert("RGB")
        items = [(r["id"], dets[r["id"]], r["categoria"]) for r in res["avisos"]]
        annotate(image, items).save(dst / "anotada.jpg", quality=88)
        append_csv(out / "resultados.csv", res, backends)
        print(f"{stem}: {Counter(r['categoria'] for r in res['avisos'])}", flush=True)

    missing = len(labels) - applied
    print(f"\nEtiquetas aplicadas: {applied}" + (f" (no encontradas: {missing})" if missing else ""))
    print(f"Cambiaron de categoría: {sum(changed.values())}")
    for (m, h), n in changed.most_common():
        print(f"  {m:13} → {h:13} {n}")


if __name__ == "__main__":
    main()
