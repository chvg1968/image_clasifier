"""Ejemplos de referencia (assets/images_examples) que se muestran al clasificador.

La etiqueta y la lección de cada ejemplo salen del nombre del archivo:
    ej_valla_*        → Valla
    ej_fachada_*      → Fachada
    ej_borde_*        → caso de borde (ver BORDE)
    ej_no_dibujar_*   → NoPublicidad: no se marca
    ej_no_dudoso_*    → NO es Dudoso aunque esté parcialmente tapado
Se usan las imágenes de contexto (sin el sufijo _detalle), que traen el
aviso enmarcado.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "assets" / "images_examples"

# Casos de borde decididos por el equipo, por palabra clave del nombre.
BORDE = {
    "totem": ("Valla", "Tótem publicitario sobre su propio poste o estructura (aunque esté junto a un negocio o sala de ventas) se clasifica como Valla."),
}

# Lección específica por palabra clave; el resto usa una lección genérica por categoría.
NOTAS = {
    "parqueadero": "Aviso pintado o adosado en una columna o muro del establecimiento → Fachada.",
    "estacion_tapada": "Aviso de fachada (marquesina de estación de servicio) parcialmente tapado por un árbol: se identifica la marca, así que es Fachada, no Dudoso.",
    "directorio": "Directorio con varios avisos de negocios sobre la fachada de un edificio → un solo aviso Fachada.",
    "sala_ventas": "Pendón o aviso instalado sobre la fachada de una sala de ventas → Fachada.",
    "contenedor_camion": "Marca impresa en un contenedor de camión: publicidad en vehículos NO se marca.",
    "tanque_camion": "Marca en el tanque de un camión: publicidad en vehículos NO se marca.",
    "furgon": "Marca en un furgón: publicidad en vehículos NO se marca.",
    "bus": "Rótulo de empresa en un bus: publicidad en vehículos NO se marca.",
    "salida": "Señal informativa ('SALIDA') sin marca comercial: no es publicidad, NO se marca.",
    "tapado_semaforo": "Aviso de fachada tapado en parte por un semáforo: aún se lee la marca, así que es Fachada, no Dudoso.",
    "cantina_tapada": "Aviso de fachada tapado en parte por un contenedor: aún se lee, así que es Fachada, no Dudoso.",
    "kenworth": "Aviso de gran formato sobre estructura metálica propia (monoposte) → Valla.",
}


@dataclass
class Example:
    path: Path
    label: str  # Valla | Fachada | NoPublicidad
    note: str

    def caption(self) -> str:
        return f"Ejemplo de referencia — categoría correcta: {self.label}. {self.note}"


def _label_and_note(stem: str) -> tuple[str, str] | None:
    note = next((n for k, n in NOTAS.items() if k in stem), "")
    if stem.startswith("ej_valla"):
        return "Valla", note
    if stem.startswith("ej_fachada"):
        return "Fachada", note
    if stem.startswith("ej_no_dibujar"):
        return "NoPublicidad", note
    if stem.startswith("ej_no_dudoso"):
        return "Fachada", note
    if stem.startswith("ej_borde"):
        for k, (label, n) in BORDE.items():
            if k in stem:
                return label, n
    return None


def load_examples(directory: Path = EXAMPLES_DIR) -> list[Example]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or re.search(r"_detalle$", p.stem):
            continue
        ln = _label_and_note(p.stem)
        if ln:
            out.append(Example(p, *ln))
    return out


def example_image(ex: Example, max_side: int = 1000) -> Image.Image:
    img = Image.open(ex.path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img
