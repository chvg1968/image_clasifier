"""Interfaz web: python -m app.ui  →  http://127.0.0.1:7860"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from .classify import build_classifiers
from .pipeline import iter_images, process_image
from .segment import DEFAULT_PROMPTS, build_segmenter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
ASSETS = ROOT / "assets" / "images"
OUTPUT = ROOT / "output_ui"


@lru_cache(maxsize=1)
def _segmenter(name: str):
    return build_segmenter(name)


def run(image_path, segmenter_name, classifier_name, threshold, prompts):
    if not image_path:
        raise gr.Error("Sube una imagen o elige una de assets/images.")
    try:
        classifiers = build_classifiers(classifier_name)
        segmenter = _segmenter(segmenter_name)
    except RuntimeError as e:
        raise gr.Error(str(e)) from e

    plist = [p.strip() for p in prompts.split(",") if p.strip()] or None
    path = Path(image_path)
    res = process_image(path, segmenter, classifiers, OUTPUT, plist, box_threshold=threshold)
    backends = [c.name for c in classifiers]
    rows = []
    for a in res["avisos"] + res["descartados"]:
        cls = a["clasificacion"]
        rows.append(
            [a["id"], a["categoria"]]
            + [cls[b]["label"] for b in backends]
            + [cls[backends[0]]["texto"], cls[backends[0]]["razon"], str([int(v) for v in a["bbox_xyxy"]])]
        )
    out_dir = OUTPUT / path.stem
    headers = ["id", "categoría"] + backends + ["texto", "razón", "bbox"]
    return str(out_dir / "anotada.jpg"), gr.Dataframe(value=rows, headers=headers), res, str(out_dir / "resultado.json")


def build() -> gr.Blocks:
    examples = [[str(p)] for p in iter_images(ASSETS)][:12] if ASSETS.exists() else []
    with gr.Blocks(title="Clasificador de avisos") as demo:
        gr.Markdown("## Clasificador de avisos publicitarios\nValla · Fachada · Dudoso, con máscara por aviso. Acepta panorámicas 360°.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="filepath", label="Imagen")
                seg = gr.Dropdown(["sam3", "dino-sam2"], value="sam3", label="Segmentador")
                cls = gr.Dropdown(["haiku", "openai", "both", "none"], value="haiku", label="Clasificador")
                thr = gr.Slider(0.1, 0.8, value=0.4, step=0.01, label="Umbral de detección (SAM 3: 0.4 · DINO: 0.25)")
                prm = gr.Textbox(value=", ".join(DEFAULT_PROMPTS), label="Frases del detector (inglés, separadas por coma)")
                btn = gr.Button("Procesar", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=[inp], label="assets/images")
            with gr.Column(scale=2):
                out_img = gr.Image(label="Resultado")
                table = gr.Dataframe(label="Avisos", wrap=True)
                with gr.Accordion("JSON", open=False):
                    js = gr.JSON()
                dl = gr.File(label="Descargar resultado.json")
        btn.click(run, [inp, seg, cls, thr, prm], [out_img, table, js, dl])
    return demo


if __name__ == "__main__":
    build().launch(allowed_paths=[str(OUTPUT), str(ASSETS)])
