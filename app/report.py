"""Página de revisión: python -m app.report [output]  →  output/revision.html

Muestra cada objeto detectado con su recorte y la etiqueta de cada clasificador.
Los desacuerdos aparecen primero, que es donde más vale la revisión humana.
"""

from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--card:#f5f5f7;--line:#d2d2d7;--warn:#fff4e5}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#f5f5f7;--muted:#a1a1a6;--card:#1c1c1e;--line:#38383a;--warn:#3a2a10}}
body{font:14px/1.45 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}
.muted{color:var(--muted)}table.sum{border-collapse:collapse;margin:8px 0}
table.sum td,table.sum th{border:1px solid var(--line);padding:4px 10px;text-align:right}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.card.diff{background:var(--warn)}
.card img{width:100%;height:180px;object-fit:contain;background:#0001;border-radius:6px}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-weight:600;font-size:12px;color:#000;margin-right:4px}
.Valla{background:#00aaff}.Fachada{background:#ffaa00}.Dudoso{background:#dc00dc;color:#fff}.NoPublicidad{background:#999}
.small{font-size:12px}a{color:inherit}
"""


def tag(label: str) -> str:
    return f'<span class="tag {html.escape(label)}">{html.escape(label)}</span>'


def build(out_dir: Path) -> Path:
    items, filtered = [], 0
    for rj in sorted(out_dir.glob("*/resultado.json")):
        res = json.loads(rj.read_text())
        stem = rj.parent.name
        for r in res["avisos"] + res["descartados"]:
            if any(c.get("razon", "").startswith("filtrado") for c in r["clasificacion"].values()):
                filtered += 1
                continue
            items.append((stem, r))
    if not items:
        raise SystemExit(f"No hay resultados en {out_dir}")

    backends = sorted({b for _, r in items for b in r["clasificacion"]})
    agree = sum(len({r["clasificacion"][b]["label"] for b in backends}) == 1 for _, r in items)
    counts = {b: Counter(r["clasificacion"][b]["label"] for _, r in items) for b in backends}
    counts["final"] = Counter(r["categoria"] for _, r in items)
    labels = ["Valla", "Fachada", "Dudoso", "NoPublicidad"]

    sum_rows = "".join(
        f"<tr><th>{b}</th>" + "".join(f"<td>{counts[b].get(l, 0)}</td>" for l in labels) + "</tr>" for b in backends + ["final"]
    )

    def card(stem, r):
        labs = {b: r["clasificacion"][b]["label"] for b in backends}
        diff = len(set(labs.values())) > 1
        body = "".join(
            f'<div class="small"><b>{b}</b> {tag(labs[b])} {html.escape(r["clasificacion"][b].get("texto", "")[:60])}'
            f'<br><span class="muted">{html.escape(r["clasificacion"][b].get("razon", "")[:220])}</span></div>'
            for b in backends
        )
        return (
            f'<div class="card{" diff" if diff else ""}"><a href="{stem}/anotada.jpg"><img loading="lazy" src="{stem}/{r["recorte"]}"></a>'
            f'<div class="small">final {tag(r["categoria"])}</div>'
            f'<div class="small muted">{stem} · #{r["id"]} · det {r["deteccion"]["score"]}</div>{body}</div>'
        )

    ordered = sorted(items, key=lambda it: len({it[1]["clasificacion"][b]["label"] for b in backends}) == 1)
    n_diff = len(items) - agree
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Revisión de avisos</title><style>{CSS}</style></head><body>
<h1>Revisión de avisos</h1>
<div class="muted">{len({s for s, _ in items})} imágenes · {len(items)} objetos clasificados ({filtered} más filtrados por tamaño o score) · acuerdo entre clasificadores: {agree}/{len(items)} ({100 * agree / len(items):.0f}%)</div>
<table class="sum"><tr><th></th>{"".join(f"<th>{l}</th>" for l in labels)}</tr>{sum_rows}</table>
<h2>Desacuerdos ({n_diff}) y luego el resto</h2>
<div class="grid">{"".join(card(s, r) for s, r in ordered)}</div>
</body></html>"""
    path = out_dir / "revision.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "output")))
