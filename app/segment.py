"""Detección y segmentación de avisos publicitarios.

Dos backends:
- "sam3" (por defecto): SAM 3 detecta y segmenta directamente desde texto.
  Requiere licencia aceptada en https://huggingface.co/facebook/sam3 y HF_TOKEN.
- "dino-sam2": Grounding DINO encuentra cajas a partir de texto y SAM 2.1
  convierte cada caja en una máscara. Ambos modelos son públicos.

Las panorámicas 360° (equirectangulares, ~12K px) se procesan en mosaicos a
resolución completa: a escala reducida los avisos quedan de pocos píxeles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

Image.MAX_IMAGE_PIXELS = None  # panorámicas de 75 MP

DEFAULT_PROMPTS = [
    "sign",
    "store sign",
    "billboard",
    "advertisement",
    "banner",
    "poster",
]


@dataclass
class Detection:
    box: list[float]  # [x0, y0, x1, y1] en píxeles de la imagen completa
    score: float
    prompt: str
    # Máscara binaria recortada a la caja; mask_xy es su esquina superior izquierda.
    mask: np.ndarray | None = field(default=None, repr=False)
    mask_xy: tuple[int, int] = (0, 0)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_panorama(image: Image.Image) -> bool:
    return image.width >= 4000 and abs(image.width / image.height - 2.0) < 0.05


def tile_grid(
    w: int, h: int, tile: int, overlap: float, band: tuple[float, float] = (0.0, 1.0)
) -> list[tuple[int, int, int, int]]:
    """Mosaicos cuadrados con solape, limitados a una franja vertical."""
    y_lo, y_hi = int(band[0] * h), int(band[1] * h)
    stride = int(tile * (1 - overlap))

    def starts(lo: int, hi: int) -> list[int]:
        if hi - lo <= tile:
            return [lo]
        s = list(range(lo, hi - tile, stride))
        s.append(hi - tile)
        return s

    return [(x, y, x + tile, y + tile) for y in starts(y_lo, y_hi) for x in starts(0, w)]


def _crop_mask(full: np.ndarray, box: list[float]) -> tuple[np.ndarray, tuple[int, int]]:
    """Recorta una máscara del tamaño del mosaico a la caja de la detección."""
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(full.shape[1], int(np.ceil(box[2]))), min(full.shape[0], int(np.ceil(box[3])))
    return full[y0:y1, x0:x1].copy(), (x0, y0)


def dedupe(dets: list[Detection], iou: float = 0.5, contain: float = 0.8) -> list[Detection]:
    """NMS global + resuelve cajas casi contenidas en otra: el mismo aviso cortado
    por el borde de un mosaico, o detectado entero en la pasada gruesa. Si la caja
    grande tiene un score parecido, gana la grande (el aviso completo)."""
    if not dets:
        return []
    boxes = torch.tensor([d.box for d in dets], dtype=torch.float32)
    scores = torch.tensor([d.score for d in dets], dtype=torch.float32)
    kept = [dets[i] for i in nms(boxes, scores, iou).tolist()]

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    result: list[Detection] = []
    for d in kept:  # ya vienen ordenadas por score descendente
        dup = False
        for j, o in enumerate(result):
            ix0, iy0 = max(d.box[0], o.box[0]), max(d.box[1], o.box[1])
            ix1, iy1 = min(d.box[2], o.box[2]), min(d.box[3], o.box[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if inter / max(1.0, min(area(d.box), area(o.box))) > contain:
                dup = True
                if area(d.box) > 1.5 * area(o.box) and d.score >= o.score - 0.15:
                    result[j] = d
                break
        if not dup:
            result.append(d)
    return result


class _Segmenter:
    """Base: aplica detect_region a la imagen completa o a sus mosaicos."""

    tile = 2048
    overlap = 0.25
    band = (0.18, 0.70)  # franja útil de la panorámica: sin cielo cenital ni capó del vehículo
    coarse_scale = 0.4  # segunda pasada reducida para avisos más grandes que un mosaico

    def detect(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
        box_threshold: float | None = None,
        min_side: int = 12,
        max_area_frac: float = 0.5,
    ) -> list[Detection]:
        prompts = prompts or DEFAULT_PROMPTS
        if is_panorama(image):
            passes = [(image, 1.0), (image.resize((int(image.width * self.coarse_scale), int(image.height * self.coarse_scale)), Image.LANCZOS), self.coarse_scale)]
        else:
            passes = [(image, 1.0)]

        dets: list[Detection] = []
        for img, scale in passes:
            if is_panorama(image):
                regions = tile_grid(img.width, img.height, self.tile, self.overlap, self.band)
            else:
                regions = [(0, 0, img.width, img.height)]
            for rx0, ry0, rx1, ry1 in regions:
                region = img.crop((rx0, ry0, rx1, ry1))
                max_area = max_area_frac * region.width * region.height
                for d in self.detect_region(region, prompts, box_threshold):
                    if (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) > max_area:
                        continue
                    d.box = [(d.box[0] + rx0) / scale, (d.box[1] + ry0) / scale, (d.box[2] + rx0) / scale, (d.box[3] + ry0) / scale]
                    d.mask_xy = (int((d.mask_xy[0] + rx0) / scale), int((d.mask_xy[1] + ry0) / scale))
                    if d.mask is not None and scale != 1.0:
                        h, w = d.mask.shape
                        d.mask = cv2.resize(d.mask.astype(np.uint8), (max(1, round(w / scale)), max(1, round(h / scale))), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if min(d.box[2] - d.box[0], d.box[3] - d.box[1]) >= min_side:
                        dets.append(d)

        dets = dedupe(dets)
        self.finalize(image, dets)
        return dets

    def detect_region(self, region: Image.Image, prompts: list[str], box_threshold: float | None) -> list[Detection]:
        raise NotImplementedError

    def finalize(self, image: Image.Image, dets: list[Detection]) -> None:
        """Paso opcional sobre la imagen completa (p. ej. calcular máscaras)."""


class Sam3Segmenter(_Segmenter):
    def __init__(self, model_id: str = "facebook/sam3", device: str | None = None):
        from transformers import Sam3Model, Sam3Processor

        if not os.getenv("HF_TOKEN"):
            raise RuntimeError(
                "SAM 3 requiere acceso aprobado en https://huggingface.co/facebook/sam3 "
                "y la variable HF_TOKEN en .env. Usa --segmenter dino-sam2 mientras tanto."
            )
        self.device = device or pick_device()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()
        self._text_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _text_inputs(self, prompt: str) -> dict[str, torch.Tensor]:
        # input_ids + attention_mask: sin la máscara, el relleno del texto contamina la consulta.
        if prompt not in self._text_cache:
            t = self.processor(text=prompt, return_tensors="pt")
            self._text_cache[prompt] = {"input_ids": t.input_ids.to(self.device), "attention_mask": t.attention_mask.to(self.device)}
        return self._text_cache[prompt]

    @torch.inference_mode()
    def detect_region(self, region, prompts, box_threshold):
        th = 0.4 if box_threshold is None else box_threshold
        pix = self.processor(images=region, return_tensors="pt").pixel_values.to(self.device)
        vision_embeds = self.model.get_vision_features(pixel_values=pix)
        out: list[Detection] = []
        for p in prompts:
            outputs = self.model(vision_embeds=vision_embeds, **self._text_inputs(p))
            res = self.processor.post_process_instance_segmentation(
                outputs, threshold=th, mask_threshold=0.5, target_sizes=[region.size[::-1]]
            )[0]
            for b, s, m in zip(res["boxes"].cpu(), res["scores"].cpu(), res["masks"].cpu()):
                box = [float(v) for v in b]
                mask, xy = _crop_mask(m.numpy().astype(bool), box)
                out.append(Detection(box, float(s), p, mask, xy))
        return out


class DinoSam2Segmenter(_Segmenter):
    def __init__(
        self,
        detector_id: str = "IDEA-Research/grounding-dino-base",
        sam_id: str = "facebook/sam2.1-hiera-large",
        device: str | None = None,
    ):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self.device = device or pick_device()
        self.det_processor = AutoProcessor.from_pretrained(detector_id)
        self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(sam_id)
        self.sam_model = Sam2Model.from_pretrained(sam_id).to(self.device).eval()

    @torch.inference_mode()
    def detect_region(self, region, prompts, box_threshold):
        th = 0.25 if box_threshold is None else box_threshold
        text = ". ".join(p.lower().strip() for p in prompts) + "."
        inputs = self.det_processor(images=region, text=text, return_tensors="pt").to(self.device)
        outputs = self.det_model(**inputs)
        res = self.det_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=th, text_threshold=0.2, target_sizes=[region.size[::-1]]
        )[0]
        return [
            Detection([float(v) for v in b], float(s), str(lbl))
            for b, s, lbl in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"])
        ]

    @torch.inference_mode()
    def finalize(self, image, dets):
        # SAM 2 sobre una ventana alrededor de cada caja, para no perder resolución.
        for d in dets:
            x0, y0, x1, y1 = d.box
            m = max(x1 - x0, y1 - y0) * 0.5 + 32
            wx0, wy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
            wx1, wy1 = min(image.width, int(x1 + m)), min(image.height, int(y1 + m))
            win = image.crop((wx0, wy0, wx1, wy1))
            box = [[[x0 - wx0, y0 - wy0, x1 - wx0, y1 - wy0]]]
            inputs = self.sam_processor(images=win, input_boxes=box, return_tensors="pt").to(self.device)
            outputs = self.sam_model(**inputs, multimask_output=False)
            full = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
            local, (lx, ly) = _crop_mask(full[0, 0].numpy().astype(bool), box[0][0])
            d.mask, d.mask_xy = local, (lx + wx0, ly + wy0)


def build_segmenter(name: str):
    if name == "sam3":
        return Sam3Segmenter()
    if name == "dino-sam2":
        return DinoSam2Segmenter()
    raise ValueError(f"Segmentador desconocido: {name}")
