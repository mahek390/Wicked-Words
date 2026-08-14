"""
Local Vision Model — Florence-2
=================================
Replaces all Gemini API calls with a local Florence-2 model.
Runs on Apple Silicon MPS (M-series) with zero cost and no rate limits.

Florence-2 tasks used:
  CAPTION          — one-sentence scene description
  DENSE_REGION_CAPTION — detect + label all objects with bboxes
  OPEN_VOCABULARY_DETECTION (<OVD>) — detect specific objects by text query
  OCR_WITH_REGION  — read text and return bounding boxes
"""

import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

log = logging.getLogger(__name__)

MODEL_ID = "microsoft/Florence-2-large"
_model = None
_processor = None


def _get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor
    device = _get_device()
    log.info(f"Loading Florence-2 on {device} (first run downloads ~800MB)...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    _model.eval()
    log.info("Florence-2 loaded.")
    return _model, _processor


def run_task(image: Image.Image, task: str, text_input: str = "") -> dict:
    """
    Run a Florence-2 task on an image.
    Returns the parsed result dict from the processor.
    """
    model, processor = load_model()
    device = next(model.parameters()).device

    prompt = task if not text_input else f"{task}{text_input}"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
        )

    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(text, task=task, image_size=image.size)
    return result


def detect_object(image: Image.Image, query: str) -> list[dict]:
    """
    Open-vocabulary detection: find objects matching `query`.
    Returns list of {label, bbox_frac: [x1,y1,x2,y2]} sorted by confidence proxy (area).
    """
    W, H = image.size
    result = run_task(image, "<OPEN_VOCABULARY_DETECTION>", query)
    out = result.get("<OPEN_VOCABULARY_DETECTION>", {})
    bboxes = out.get("bboxes", [])
    labels = out.get("bboxes_labels", [])

    detections = []
    total_area = W * H
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        # Normalized area fraction capped at 1.0 — larger detections are more
        # likely to be the primary object Florence-2 matched to the query.
        area_frac = min(area / total_area, 1.0)
        detections.append({
            "label": label,
            "bbox_frac": [round(x1/W, 4), round(y1/H, 4),
                          round(x2/W, 4), round(y2/H, 4)],
            "bbox_px": [x1, y1, x2, y2],
            "area": area,
            "area_frac": round(area_frac, 4),
        })
    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


def dense_caption(image: Image.Image) -> list[dict]:
    """
    Dense region captioning: returns all detected regions with labels and bboxes.
    """
    W, H = image.size
    result = run_task(image, "<DENSE_REGION_CAPTION>")
    out = result.get("<DENSE_REGION_CAPTION>", {})
    bboxes = out.get("bboxes", [])
    labels = out.get("labels", [])

    regions = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        regions.append({
            "label": label,
            "bbox_frac": [round(x1/W, 4), round(y1/H, 4),
                          round(x2/W, 4), round(y2/H, 4)],
            "bbox_px": [x1, y1, x2, y2],
            "area": (x2 - x1) * (y2 - y1),
        })
    return regions


def caption(image: Image.Image) -> str:
    """Single-sentence scene description."""
    result = run_task(image, "<CAPTION>")
    return result.get("<CAPTION>", "").strip()


def ocr_with_regions(image: Image.Image) -> list[dict]:
    """
    OCR with bounding boxes. Returns list of {text, bbox_frac}.
    """
    W, H = image.size
    result = run_task(image, "<OCR_WITH_REGION>")
    out = result.get("<OCR_WITH_REGION>", {})
    quads = out.get("quad_boxes", [])
    labels = out.get("labels", [])

    regions = []
    for quad, label in zip(quads, labels):
        # quad is [x1,y1,x2,y1,x2,y2,x1,y2] — convert to bbox
        xs = quad[0::2]
        ys = quad[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        regions.append({
            "text": label,
            "bbox_frac": [round(x1/W, 4), round(y1/H, 4),
                          round(x2/W, 4), round(y2/H, 4)],
            "height_px": y2 - y1,
        })
    return regions