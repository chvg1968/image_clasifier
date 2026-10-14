"""Une segmentación + clasificación y genera las salidas por imagen."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .classify import Classification, crop_object
from .segment import Detection

COLORS = {
    "Valla": (0, 170, 255),
    "Fachada": (255, 170, 0),
    "Dudoso": (220, 0, 220),
    "SinClasificar": (0, 220, 120),
    "NoPublicidad": (150, 150, 150),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ANNOTATED_MAX_WIDTH = 6144  # la anotada de una panorámica 12K se guarda a la mitad


def mask_to_polygon(mask: np.ndarray, origin: tuple[int, int] = (0, 0), epsilon_frac: float = 0.004) -> list[list[int]]:
    """Contorno externo más grande de la máscara, simplificado, en coordenadas de la imagen."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(1.0, epsilon_frac * cv2.arcLength(c, True)), True).reshape(-1, 2)
    return (c + np.array(origin)).astype(int).tolist()


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image: Image.Image, items: list[tuple[int, Detection, str]], max_width: int = ANNOTATED_MAX_WIDTH) -> Image.Image:
    """items: (id, detección, categoría). Dibuja máscara, contorno, caja y etiqueta."""
    s = min(1.0, max_width / image.width)
    out = image.convert("RGB").resize((int(image.width * s), int(image.height * s)), Image.LANCZOS) if s < 1 else image.convert("RGB").copy()
    arr = np.array(out).astype(np.float32)
    for _, det, label in items:
        if det.mask is None or not det.mask.any():
            continue
        mh, mw = det.mask.shape
        x0, y0 = int(det.mask_xy[0] * s), int(det.mask_xy[1] * s)
        m = cv2.resize(det.mask.astype(np.uint8), (max(1, int(mw * s)), max(1, int(mh * s))), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = m[: arr.shape[0] - y0, : arr.shape[1] - x0]
        region = arr[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]]
        region[m] = 0.55 * region[m] + 0.45 * np.array(COLORS.get(label, (255, 255, 255)), dtype=np.float32)
    out = Image.fromarray(arr.astype(np.uint8))

    draw = ImageDraw.Draw(out)
    lw = max(2, int(max(out.size) / 1500))
    font = _font(max(14, int(max(out.size) / 220)))
    for i, det, label in items:
        color = COLORS.get(label, (255, 255, 255))
        if det.mask is not None:
            poly = [(x * s, y * s) for x, y in mask_to_polygon(det.mask, det.mask_xy)]
            if len(poly) >= 3:
                draw.line(poly + [poly[0]], fill=color, width=lw)
        x0, y0, x1, y1 = (v * s for v in det.box)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(1, lw // 2))
        tag = f"#{i} {label}"
        tb = draw.textbbox((0, 0), tag, font=font)
        ty = max(0, y0 - (tb[3] - tb[1]) - 6)
        draw.rectangle([x0, ty, x0 + tb[2] - tb[0] + 8, ty + tb[3] - tb[1] + 6], fill=color)
        draw.text((x0 + 4, ty + 1), tag, fill=(0, 0, 0), font=font)
    return out


def _classify_all(image: Image.Image, dets: list[Detection], classifiers: list, workers: int) -> list[dict[str, Classification]]:
    jobs = [(i, c) for i in range(len(dets)) for c in classifiers]

    def run(job):
        i, c = job
        try:
            return i, c.name, c.classify(image, dets[i].box)
        except Exception as e:  # un error de API no debe tumbar el lote
            return i, c.name, Classification("Dudoso", 0.5, None, "", f"error: {type(e).__name__}: {e}"[:300], backend=c.name)

    results: list[dict[str, Classification]] = [{} for _ in dets]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, name, cls in pool.map(run, jobs):
            results[i][name] = cls
    return results


def combine(results: dict[str, Classification], primary: str, min_prob: float = 0.8) -> str:
    """Categoría final. Con un solo clasificador, la suya. Con 'both', la de Haiku,
    salvo que Haiku diga Dudoso y OpenAI esté seguro de Fachada/Valla: en la revisión
    de desacuerdos Haiku dejaba en Dudoso letreros de local legibles."""
    label = results[primary].label
    other = results.get("openai")
    if primary == "haiku" and other is not None and label == "Dudoso" and other.label in ("Fachada", "Valla"):
        if other.probabilities.get(other.label, 0.0) >= min_prob:
            return other.label
    return label


def split_by_quality(dets: list[Detection], min_size: int = 60, min_score: float = 0.5) -> tuple[list[Detection], list[Detection]]:
    """Separa las detecciones demasiado pequeñas (lado mayor en px) o de bajo score.
    En panorámicas 12K casi todas son manchas imposibles de identificar."""
    keep, drop = [], []
    for d in dets:
        size = max(d.box[2] - d.box[0], d.box[3] - d.box[1])
        (keep if size >= min_size and d.score >= min_score else drop).append(d)
    return keep, drop


def process_image(
    path: Path,
    segmenter,
    classifiers: list,
    out_dir: Path,
    prompts: list[str] | None = None,
    workers: int = 8,
    min_size: int = 60,
    min_score: float = 0.5,
    **detect_kwargs,
) -> dict:
    image = Image.open(path).convert("RGB")
    dets = segmenter.detect(image, prompts=prompts, **detect_kwargs)
    keep, drop = split_by_quality(dets, min_size, min_score)
    return write_outputs(path, image, keep, classifiers, out_dir, workers, filtered=drop, min_size=min_size, min_score=min_score)


def write_outputs(
    path: Path,
    image: Image.Image,
    dets: list[Detection],
    classifiers: list,
    out_dir: Path,
    workers: int = 8,
    filtered: list[Detection] | None = None,
    min_size: int = 60,
    min_score: float = 0.5,
) -> dict:
    """Clasifica las detecciones y escribe anotada.jpg, máscaras, recortes y resultado.json."""
    per_det = _classify_all(image, dets, classifiers, workers)
    primary = classifiers[0].name
    filtered = filtered or []
    motivo = f"filtrado: lado mayor < {min_size} px o score de detección < {min_score}"
    for _ in filtered:
        per_det.append({c.name: Classification("NoPublicidad", 0.0, None, "", motivo, backend=c.name) for c in classifiers})
    dets = dets + filtered

    stem_dir = out_dir / path.stem
    (stem_dir / "masks").mkdir(parents=True, exist_ok=True)
    (stem_dir / "crops").mkdir(parents=True, exist_ok=True)

    records = []
    for i, (det, results) in enumerate(zip(dets, per_det)):
        label = combine(results, primary)
        mask_file = None
        if det.mask is not None:
            mask_file = f"masks/{i:02d}_{label}.png"
            Image.fromarray(det.mask.astype(np.uint8) * 255).save(stem_dir / mask_file)
        crop_object(image, det.box, min_side=256).save(stem_dir / "crops" / f"{i:02d}.jpg", quality=88)
        records.append(
            {
                "id": i,
                "categoria": label,
                "bbox_xyxy": [round(v, 1) for v in det.box],
                "poligono": mask_to_polygon(det.mask, det.mask_xy) if det.mask is not None else [],
                "area_px": int(det.mask.sum()) if det.mask is not None else None,
                "mascara": mask_file,
                "mascara_origen_xy": list(det.mask_xy),
                "recorte": f"crops/{i:02d}.jpg",
                "deteccion": {"score": round(det.score, 3), "prompt": det.prompt},
                "clasificacion": {name: c.to_dict() for name, c in results.items()},
            }
        )

    items = [(r["id"], dets[r["id"]], r["categoria"]) for r in records if r["categoria"] != "NoPublicidad"]
    annotate(image, items).save(stem_dir / "anotada.jpg", quality=88)

    result = {
        "imagen": str(path),
        "ancho": image.width,
        "alto": image.height,
        "clasificador_principal": primary,
        "avisos": [r for r in records if r["categoria"] != "NoPublicidad"],
        "descartados": [r for r in records if r["categoria"] == "NoPublicidad"],
    }
    (stem_dir / "resultado.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def load_detections(result_json: Path) -> tuple[Path, list[Detection]]:
    """Reconstruye las detecciones (cajas + máscaras) de un resultado.json previo."""
    res = json.loads(result_json.read_text())
    dets = []
    for r in sorted(res["avisos"] + res["descartados"], key=lambda r: r["id"]):
        mask = None
        if r.get("mascara"):
            mask = np.array(Image.open(result_json.parent / r["mascara"])) > 127
        dets.append(Detection(r["bbox_xyxy"], r["deteccion"]["score"], r["deteccion"]["prompt"], mask, tuple(r["mascara_origen_xy"])))
    return Path(res["imagen"]), dets


CSV_FIELDS = ["imagen", "id", "categoria", "x0", "y0", "x1", "y1", "area_px", "score_det", "prompt", "recorte"]


def append_csv(csv_path: Path, result: dict, backends: list[str]) -> None:
    fields = CSV_FIELDS + [f"{b}_{k}" for b in backends for k in ("categoria", "texto", "razon")]
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        stem = Path(result["imagen"]).stem
        for r in result["avisos"] + result["descartados"]:
            row = {
                "imagen": stem,
                "id": r["id"],
                "categoria": r["categoria"],
                "x0": r["bbox_xyxy"][0],
                "y0": r["bbox_xyxy"][1],
                "x1": r["bbox_xyxy"][2],
                "y1": r["bbox_xyxy"][3],
                "area_px": r["area_px"],
                "score_det": r["deteccion"]["score"],
                "prompt": r["deteccion"]["prompt"],
                "recorte": f"{stem}/{r['recorte']}",
            }
            for b in backends:
                c = r["clasificacion"].get(b, {})
                row[f"{b}_categoria"] = c.get("label", "")
                row[f"{b}_texto"] = c.get("texto", "")
                row[f"{b}_razon"] = c.get("razon", "")
            w.writerow(row)


def iter_images(target: Path):
    if target.is_file():
        yield target
        return
    for p in sorted(target.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
            yield p
