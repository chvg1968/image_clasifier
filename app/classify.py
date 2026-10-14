"""Clasificación de cada aviso detectado en Valla / Fachada / Dudoso.

Cada clasificador recibe:
- Los ejemplos de referencia de assets/images_examples con su etiqueta correcta.
- El recorte del objeto detectado, para leer su contenido.
- Una ventana de contexto alrededor del objeto, con el objeto enmarcado en rojo,
  para ver la estructura (poste propio vs. pared de un edificio vs. vehículo).
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import asdict, dataclass, field

from PIL import Image, ImageDraw

from .examples import example_image, load_examples

LABELS = ["Valla", "Fachada", "Dudoso"]

CRITERIOS = """\
Las fotos son recortes de panorámicas 360° tomadas desde un vehículo en calles de Colombia.

Categorías:
- Valla: aviso publicitario sobre una estructura propia e independiente (monoposte, \
estructura metálica, valla en azotea o en lote, tótem sobre su propio poste), aunque \
esté junto a un negocio.
- Fachada: aviso adosado, pintado o instalado sobre la pared, columna, muro, marquesina, \
toldo o vitrina de un edificio o local (letrero principal del negocio, pendones \
comerciales en fachada, directorios de locales).
- Dudoso: el objeto muestra alguna señal comercial (marca, logo, nombre de negocio, teléfono, \
producto, "se vende/arrienda", panel típico de aviso) pero hay ambigüedad real sobre si es \
publicidad, o se sabe que es un aviso pero su contenido no se puede leer (borroso, lejano o \
casi todo tapado). Dudoso debe ser poco frecuente.

Reglas:
- Un aviso parcialmente tapado (por un árbol, un semáforo, un vehículo) cuyo contenido o \
marca todavía se identifica NO es Dudoso: clasifícalo por su ubicación.
- Un objeto tan pequeño, oscuro o borroso que no se puede saber qué es, y sin ninguna señal \
comercial visible, NO es publicidad (no se marca): es ruido de la detección.
- NO son publicidad (no se marcan): marcas o rótulos en vehículos (buses, camiones, \
contenedores, furgones, taxis), señales de tránsito y nomenclatura vial, señales \
informativas sin marca (SALIDA, PARE, números de casa), semáforos, grafitis sin marca, \
ventanas, rejas y objetos que solo se parecen a un aviso.
- Decisiones del equipo, también NO son publicidad (NoPublicidad):
  * carteles particulares de "se vende" / "se arrienda" pegados en la fachada o la reja;
  * el nombre del propio edificio, conjunto o bloque en su muro (placas "BLOQUE 7", "TORRE B", \
"Conjunto Los Pinos", letras corpóreas con el nombre del edificio). En cambio, un \
proyecto inmobiliario anunciado en una valla SÍ es Valla;
  * letreros secundarios o informativos de un local ("servicio a domicilio", "garaje en uso", \
"propiedad privada", horarios), calcomanías de vitrina y el poste de barbería: solo cuenta \
el aviso principal del negocio o una marca anunciada;
  * letreros de canchas y escenarios deportivos del barrio, aunque muestren un patrocinador."""


@dataclass
class Classification:
    label: str  # Valla | Fachada | Dudoso | NoPublicidad
    es_publicidad: float  # probabilidad / certeza 0..1
    legible: bool | None = None
    texto: str = ""
    razon: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    backend: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Lado mayor de la ventana de contexto enviada a la API. 768 px cuesta ~4× menos tokens que 1568
# y en las etiquetas manuales rindió igual o mejor.
CONTEXT_MAX_SIDE = int(os.getenv("CONTEXT_MAX_SIDE", "768"))


def _b64(img: Image.Image, max_side: int = 1568) -> str:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def crop_object(image: Image.Image, box: list[float], pad: float = 0.25, min_side: int = 384) -> Image.Image:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    crop = image.crop(
        (
            max(0, int(x0 - pad * w)),
            max(0, int(y0 - pad * h)),
            min(image.width, int(x1 + pad * w)),
            min(image.height, int(y1 + pad * h)),
        )
    )
    if max(crop.size) < min_side:  # recortes diminutos se amplían para poder leerlos
        s = min_side / max(crop.size)
        crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.LANCZOS)
    return crop


def context_window(image: Image.Image, box: list[float], factor: float = 4.0, min_side: int = 1200) -> Image.Image:
    """Ventana alrededor del objeto con el objeto enmarcado en rojo."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(factor * max(x1 - x0, y1 - y0), min_side) / 2
    wx0, wy0 = max(0, int(cx - half)), max(0, int(cy - half))
    wx1, wy1 = min(image.width, int(cx + half)), min(image.height, int(cy + half))
    win = image.crop((wx0, wy0, wx1, wy1)).convert("RGB")
    draw = ImageDraw.Draw(win)
    lw = max(3, int(max(win.size) / 250))
    draw.rectangle([x0 - wx0 - lw, y0 - wy0 - lw, x1 - wx0 + lw, y1 - wy0 + lw], outline=(255, 0, 0), width=lw)
    return win


def _decide(es_pub: float, label: str, legible: bool | None, no_pub_below: float, dudoso_below: float) -> str:
    """es_pub < no_pub_below → claramente no es publicidad (se descarta).
    no_pub_below ≤ es_pub < dudoso_below → ambigüedad sobre si es aviso → Dudoso."""
    if es_pub < no_pub_below:
        return "NoPublicidad"
    if es_pub < dudoso_below:
        return "Dudoso"
    if legible is False and label != "Valla":  # una valla se reconoce por su estructura aunque no se lea
        return "Dudoso"
    return label


PREGUNTA = (
    "¿El objeto enmarcado en rojo es un aviso publicitario? Usa es_publicidad=true solo si ves "
    "una señal comercial positiva (marca, logo, nombre de negocio, teléfono, producto, 'se vende/"
    "arrienda', o una estructura típica de aviso como el marco de una valla o el panel de un letrero "
    "de local). Usa es_publicidad=false para vehículos, señales de tránsito o informativas, semáforos, "
    "grafitis sin marca, ventanas, rejas, y para objetos tan pequeños o borrosos que no se puede saber "
    "qué son (en 'razon' escribe 'no identificable'). Usa categoria='Dudoso' solo si es claramente un "
    "aviso cuyo contenido no se lee, o si hay una señal comercial pero dudas real de que sea publicidad. "
    "Marca legible=false si no se puede leer ni identificar su contenido. En 'texto' transcribe lo "
    "legible (vacío si nada). En 'razon' explica en una frase, mencionando la estructura que lo soporta."
)


# --------------------------------------------------------------------------- #
# Claude Haiku 5.5
# --------------------------------------------------------------------------- #

HAIKU_SCHEMA = {
    "type": "object",
    "properties": {
        "es_publicidad": {"type": "boolean"},
        "categoria": {"type": "string", "enum": LABELS},
        "legible": {"type": "boolean"},
        "texto": {"type": "string"},
        "razon": {"type": "string"},
    },
    "required": ["es_publicidad", "categoria", "legible", "texto", "razon"],
    "additionalProperties": False,
}


class HaikuClassifier:
    name = "haiku"

    def __init__(self, model: str = "claude-haiku-5-5", effort: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.effort = effort or os.getenv("HAIKU_EFFORT", "low")
        # Bloques de ejemplos: idénticos en cada llamada → se cachean.
        blocks: list[dict] = [{"type": "text", "text": "EJEMPLOS DE REFERENCIA (etiquetados por el equipo):"}]
        for ex in load_examples():
            blocks.append({"type": "text", "text": ex.caption()})
            blocks.append(
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(example_image(ex))}}
            )
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        self.example_blocks = blocks

    def classify(self, image: Image.Image, box: list[float]) -> Classification:
        crop_b64, ctx_b64 = _b64(crop_object(image, box)), _b64(context_window(image, box), CONTEXT_MAX_SIDE)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": HAIKU_SCHEMA}},
            system=(
                "Eres un auditor de publicidad exterior. Clasificas objetos detectados automáticamente "
                "en fotos de calle.\n\n" + CRITERIOS
            ),
            messages=[
                {
                    "role": "user",
                    "content": self.example_blocks
                    + [
                        {"type": "text", "text": "OBJETO A CLASIFICAR.\nImagen A: recorte del objeto detectado."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop_b64}},
                        {"type": "text", "text": "Imagen B: alrededores; el objeto está enmarcado en rojo."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": ctx_b64}},
                        {"type": "text", "text": PREGUNTA},
                    ],
                }
            ],
        )
        if response.stop_reason != "end_turn":
            return Classification("Dudoso", 0.5, None, "", f"stop_reason={response.stop_reason}", backend=self.name)
        data = json.loads(next(b.text for b in response.content if b.type == "text"))
        es_pub = 1.0 if data["es_publicidad"] else 0.0
        return Classification(
            label=_decide(es_pub, data["categoria"], data["legible"], 0.5, 0.5),
            es_publicidad=es_pub,
            legible=data["legible"],
            texto=data["texto"],
            razon=data["razon"],
            backend=self.name,
        )


# --------------------------------------------------------------------------- #
# OpenAI Decisions API
# --------------------------------------------------------------------------- #


class DecisionsClassifier:
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-6-luna",
        no_pub_below: float = 0.2,
        dudoso_below: float = 0.5,
        min_conf: float = 0.6,
        min_legible: float = 0.5,
    ):
        from openai import OpenAI

        self.client = OpenAI(max_retries=5)
        self.model = model
        self.no_pub_below = no_pub_below
        self.dudoso_below = dudoso_below
        self.min_conf = min_conf
        self.min_legible = min_legible
        parts: list[dict] = [{"type": "input_text", "text": CRITERIOS + "\n\nEJEMPLOS DE REFERENCIA (etiquetados por el equipo):"}]
        for ex in load_examples():
            parts.append({"type": "input_text", "text": ex.caption()})
            parts.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{_b64(example_image(ex))}"})
        self.example_parts = parts

    def classify(self, image: Image.Image, box: list[float]) -> Classification:
        crop_b64, ctx_b64 = _b64(crop_object(image, box)), _b64(context_window(image, box), CONTEXT_MAX_SIDE)
        decision = self.client.decisions.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": self.example_parts
                    + [
                        {"type": "input_text", "text": "OBJETO A CLASIFICAR.\nImagen A: recorte del objeto detectado."},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{crop_b64}", "detail": "high"},
                        {"type": "input_text", "text": "Imagen B: alrededores; el objeto está enmarcado en rojo."},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{ctx_b64}"},
                    ],
                }
            ],
            questions=[
                {
                    "name": "es_publicidad",
                    "type": "predicate",
                    "instructions": (
                        "El objeto enmarcado en rojo es un aviso publicitario o comercial que se debe marcar: muestra "
                        "una señal comercial (marca, logo, nombre de negocio, teléfono, producto, 'se vende/arrienda') o "
                        "una estructura típica de aviso. Falso para vehículos, señales de tránsito o informativas, "
                        "semáforos, grafitis sin marca, ventanas, rejas, y objetos tan pequeños o borrosos que no se "
                        "puede saber qué son."
                    ),
                },
                {
                    "name": "categoria",
                    "type": "choice",
                    "instructions": "Clasifica el objeto enmarcado en rojo según las categorías y reglas descritas y los ejemplos.",
                    "choices": [
                        {"value": "Valla", "description": "Aviso sobre estructura propia e independiente (monoposte, estructura metálica, azotea, tótem en su propio poste)."},
                        {"value": "Fachada", "description": "Aviso adosado, pintado o instalado en la pared, columna, marquesina o frente de un edificio o local."},
                        {"value": "Dudoso", "description": "Tiene señal comercial pero no es claro si es publicidad, o es claramente un aviso cuyo contenido no se puede leer. Debe ser poco frecuente."},
                    ],
                },
                {
                    "name": "legible",
                    "type": "predicate",
                    "instructions": "El contenido del aviso (marca, texto o mensaje) se puede leer o identificar, aunque esté parcialmente tapado.",
                },
            ],
        )
        answers = {a.name: a for a in decision.answers}

        def prob(name: str) -> float | None:
            a = answers.get(name)
            return a.probability if a is not None and a.type == "predicate" else None

        es_pub, legible_p = prob("es_publicidad"), prob("legible")
        cat = answers.get("categoria")
        if es_pub is None or cat is None or cat.type != "choice":
            return Classification("Dudoso", es_pub or 0.5, None, "", "la API rechazó alguna pregunta", backend=self.name)

        probs = {str(p.value): round(p.probability, 4) for p in cat.probabilities}
        label = str(cat.choice)
        razon = f"P(publicidad)={es_pub:.2f}, confianza categoría={cat.confidence:.2f}"
        if legible_p is not None:
            razon += f", P(legible)={legible_p:.2f}"
        if cat.confidence < self.min_conf:
            label = "Dudoso"
        legible = None if legible_p is None else legible_p >= self.min_legible
        return Classification(
            label=_decide(es_pub, label, legible, self.no_pub_below, self.dudoso_below),
            es_publicidad=round(es_pub, 4),
            legible=legible,
            razon=razon,
            probabilities=probs,
            backend=self.name,
        )


# --------------------------------------------------------------------------- #


class NoClassifier:
    """Solo segmenta; útil para probar la detección sin gastar API."""

    name = "none"

    def classify(self, image: Image.Image, box: list[float]) -> Classification:
        return Classification("SinClasificar", 1.0, backend=self.name)


def build_classifiers(name: str) -> list:
    """'haiku' | 'openai' | 'both' | 'none'. Con 'both', el primero define la categoría final."""
    names = ["haiku", "openai"] if name == "both" else [name]
    out = []
    for n in names:
        if n == "haiku":
            if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
                raise RuntimeError("Falta ANTHROPIC_API_KEY en .env para usar el clasificador haiku.")
            out.append(HaikuClassifier())
        elif n == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Falta OPENAI_API_KEY en .env para usar el clasificador openai.")
            out.append(DecisionsClassifier())
        elif n == "none":
            out.append(NoClassifier())
        else:
            raise ValueError(f"Clasificador desconocido: {n}")
    return out
