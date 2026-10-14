"""Página para etiquetar a mano: python -m app.label_page output_v2  →  output_v2/etiquetar.html

Incluye todos los desacuerdos entre clasificadores y una muestra de acuerdos
como control. Las respuestas del modelo van ocultas por defecto para no
sesgar a quien etiqueta. Las etiquetas se guardan en el navegador y se
descargan como CSV para app.evaluate.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

from PIL import Image

from .classify import context_window

Image.MAX_IMAGE_PIXELS = None
LABELS = ["Valla", "Fachada", "Dudoso", "NoPublicidad"]

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--card:#f5f5f7;--line:#d2d2d7;--acc:#0071e3;--done:#e8f5e9}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#f5f5f7;--muted:#a1a1a6;--card:#1c1c1e;--line:#38383a;--acc:#2997ff;--done:#14301a}}
*{box-sizing:border-box}
body{font:15px/1.45 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 0;z-index:2}
h1{font-size:19px;margin:0 0 6px}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin:6px 0}
.bar>div{height:100%;background:var(--acc);width:0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{font:inherit;padding:7px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.sel{background:var(--acc);border-color:var(--acc);color:#fff}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin:14px 0;scroll-margin-top:130px}
.card.done{background:var(--done)}
.imgs{display:grid;grid-template-columns:1fr 2fr;gap:10px;margin:8px 0}
.imgs img{width:100%;max-height:340px;object-fit:contain;background:#0002;border-radius:8px}
.muted{color:var(--muted);font-size:13px}
.models{display:none;font-size:13px;margin-top:6px}
body.show .models{display:block}
@media (max-width:640px){.imgs{grid-template-columns:1fr}}
"""

JS = """
const KEY='etiquetas_'+document.title;
let store={};
try{store=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){}}
function paint(){
  const cards=[...document.querySelectorAll('.card')];
  let n=0;
  cards.forEach(c=>{
    const v=store[c.dataset.key];
    c.classList.toggle('done',!!v);
    c.querySelectorAll('button[data-l]').forEach(b=>b.classList.toggle('sel',b.dataset.l===v));
    if(v)n++;
  });
  document.getElementById('cnt').textContent=n+' / '+cards.length;
  document.getElementById('prog').style.width=(100*n/cards.length)+'%';
}
document.addEventListener('click',e=>{
  const b=e.target.closest('button[data-l]'); if(!b)return;
  const card=b.closest('.card'); store[card.dataset.key]=b.dataset.l; save(); paint();
  const next=card.nextElementSibling;
  if(next&&next.classList.contains('card')&&!store[next.dataset.key]) next.scrollIntoView({behavior:'smooth'});
});
document.addEventListener('keydown',e=>{
  const i=['1','2','3','4'].indexOf(e.key); if(i<0)return;
  const card=[...document.querySelectorAll('.card')].find(c=>!store[c.dataset.key]); if(!card)return;
  card.querySelectorAll('button[data-l]')[i].click();
});
function exportCsv(){
  const rows=[['imagen','id','grupo','etiqueta']];
  document.querySelectorAll('.card').forEach(c=>{
    if(store[c.dataset.key]) rows.push([c.dataset.img,c.dataset.id,c.dataset.grupo,store[c.dataset.key]]);
  });
  const blob=new Blob([rows.map(r=>r.join(',')).join('\\n')],{type:'text/csv'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='etiquetas_manuales.csv'; a.click();
}
function firstPending(){
  const c=[...document.querySelectorAll('.card')].find(c=>!store[c.dataset.key]); if(c)c.scrollIntoView({behavior:'smooth'});
}
function resetAll(){ if(confirm('¿Borrar todas las etiquetas de este navegador?')){store={};save();paint();} }
paint();
"""


def build(out_dir: Path, n_control: int = 30, seed: int = 7, muestra: int = 0, excluir: Path | None = None) -> Path:
    """Con muestra > 0 arma una muestra aleatoria de objetos (sin los ya etiquetados en
    `excluir`), que mide la precisión real del lote en vez de solo los casos difíciles."""
    ya = set()
    if excluir:
        import csv

        ya = {(r["imagen"], int(r["id"])) for r in csv.DictReader(open(excluir, encoding="utf-8"))}
    disagree, agree = [], []
    for rj in sorted(out_dir.glob("*/resultado.json")):
        res = json.loads(rj.read_text())
        for r in res["avisos"] + res["descartados"]:
            cls = r["clasificacion"]
            if any(c.get("razon", "").startswith("filtrado") for c in cls.values()):
                continue
            if (rj.parent.name, r["id"]) in ya:
                continue
            labs = {c["label"] for c in cls.values()}
            (disagree if len(labs) > 1 else agree).append((rj.parent.name, res["imagen"], r))

    if muestra:
        pool = disagree + agree
        items = [("aleatorio", *x) for x in random.Random(seed).sample(pool, min(muestra, len(pool)))]
    else:
        random.Random(seed).shuffle(disagree)
        control = random.Random(seed).sample(agree, min(n_control, len(agree)))
        # Desacuerdos y controles mezclados, para que quien etiqueta no sepa cuál es cuál.
        items = [("desacuerdo", *x) for x in disagree] + [("control", *x) for x in control]
        random.Random(seed + 1).shuffle(items)

    # Ventana de contexto por objeto (abre cada panorámica una sola vez).
    by_image: dict[str, list] = {}
    for it in items:
        by_image.setdefault(it[2], []).append(it)
    for img_path, its in by_image.items():
        image = Image.open(img_path).convert("RGB")
        for _, stem, _, r in its:
            ctx_dir = out_dir / stem / "ctx"
            ctx_dir.mkdir(exist_ok=True)
            win = context_window(image, r["bbox_xyxy"])
            win.thumbnail((1000, 1000))
            win.save(ctx_dir / f"{r['id']:02d}.jpg", quality=85)

    cards = []
    for n, (grupo, stem, _, r) in enumerate(items, 1):
        models = " · ".join(f"{b}: {html.escape(c['label'])}" for b, c in r["clasificacion"].items())
        buttons = "".join(f'<button data-l="{l}">{i}. {l}</button>' for i, l in enumerate(LABELS, 1))
        cards.append(
            f'<div class="card" data-key="{stem}#{r["id"]}" data-img="{stem}" data-id="{r["id"]}" data-grupo="{grupo}">'
            f'<div class="muted">{n}. {stem} · objeto #{r["id"]}</div>'
            f'<div class="imgs"><img loading="lazy" src="{stem}/{r["recorte"]}" alt="recorte">'
            f'<img loading="lazy" src="{stem}/ctx/{r["id"]:02d}.jpg" alt="contexto"></div>'
            f'<div class="row">{buttons}</div><div class="models muted">{models}</div></div>'
        )

    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Etiquetado {out_dir.name}</title><style>{CSS}</style></head><body>
<header>
<h1>Etiquetado manual de avisos</h1>
<div class="muted">Izquierda: el objeto. Derecha: sus alrededores, con el objeto en rojo. Elige la categoría correcta según los criterios del proyecto.<br><b>Regla de oro:</b> Valla o Fachada solo si lees con claridad el nombre, la marca o el mensaje, y no hay duda de que es un aviso. Si se ve que es un letrero o aviso pero no se lee, o hay duda → Dudoso (aunque sea pequeño). NoPublicidad solo si claramente no es aviso (vehículo, señal de tránsito o informativa, nomenclatura) o es una mancha irreconocible.
Atajos: teclas 1–4 etiquetan el siguiente pendiente. Lo que marques se guarda en este navegador.</div>
<div class="bar"><div id="prog"></div></div>
<div class="row"><b id="cnt"></b>
<button onclick="firstPending()">Ir al siguiente pendiente</button>
<button class="primary" onclick="exportCsv()">Descargar CSV</button>
<label class="muted"><input type="checkbox" onchange="document.body.classList.toggle('show',this.checked)"> Mostrar respuestas de los modelos (puede sesgar)</label>
<button onclick="resetAll()">Borrar todo</button></div>
</header>
{"".join(cards)}
<script>{JS}</script></body></html>"""
    path = out_dir / "etiquetar.html"
    path.write_text(page, encoding="utf-8")
    print(f"{len(items)} objetos ({'muestra aleatoria' if muestra else 'desacuerdos + controles'})")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="output_v2")
    ap.add_argument("--controles", type=int, default=30)
    ap.add_argument("--muestra", type=int, default=0, help="N objetos al azar en vez de desacuerdos")
    ap.add_argument("--excluir", type=Path, help="CSV de etiquetas ya hechas, para no repetirlas")
    a = ap.parse_args()
    print(build(Path(a.out_dir), a.controles, muestra=a.muestra, excluir=a.excluir))
