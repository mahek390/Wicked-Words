"""
Stage 3 — Target Text Detection & Extraction
=============================================
Uses Florence-2 (local, no API) to find the background text the participant
photographed, crop it, and extract metrics for Stage 4.
"""

import logging
import numpy as np
from pathlib import Path
from PIL import Image
from stage2_refcard import bbox_crop
from local_model import detect_object, dense_caption, ocr_with_regions, caption
log = logging.getLogger(__name__)

# Text-like object labels Florence-2 commonly returns
_TEXT_KEYWORDS = [
    "sign", "text", "label", "poster", "menu", "book", "newspaper",
    "package", "screen", "display", "board", "banner", "notice",
    "bottle", "box", "can", "container", "magazine", "page",
]


def refine_mask_with_sam(image_np: np.ndarray, bbox_frac: list[float], **kwargs):
    """
    Fallback stub for SAM mask refinement when SAM is disabled or not imported.
    Returns None to fallback to standard bounding-box cropping.
    """
    return None


def _is_text_region(label: str) -> bool:
    lbl = label.lower()
    return any(k in lbl for k in _TEXT_KEYWORDS)


def _dominant_color(img_crop: Image.Image, region: str = "all") -> list[int]:
    """Return median RGB of a crop (robust to outliers)."""
    arr = np.array(img_crop.convert("RGB"))
    if region == "border":
        # Sample from edges — likely background
        h, w = arr.shape[:2]
        border = np.concatenate([
            arr[:5, :].reshape(-1, 3),
            arr[-5:, :].reshape(-1, 3),
            arr[:, :5].reshape(-1, 3),
            arr[:, -5:].reshape(-1, 3),
        ])
        return np.median(border, axis=0).astype(int).tolist()
    return np.median(arr.reshape(-1, 3), axis=0).astype(int).tolist()


def _estimate_cap_height(ocr_regions: list[dict], target_bbox_frac: list[float],
                          img_size: tuple) -> float | None:
    """
    Estimate capital letter height in pixels from OCR regions that fall
    inside the target bbox.
    """
    W, H = img_size
    tx1, ty1, tx2, ty2 = target_bbox_frac
    heights = []
    for r in ocr_regions:
        rx1, ry1, rx2, ry2 = r["bbox_frac"]
        # Check overlap with target bbox
        if rx1 >= tx1 and rx2 <= tx2 and ry1 >= ty1 and ry2 <= ty2:
            heights.append(r["height_px"])
    if heights:
        return round(float(np.median(heights)), 1)
    return None


def process_image(
    response_id: str,
    image_path: Path,
    crops_dir: Path,
    card_bbox_frac: list[float] | None = None,
) -> dict:
    result = {
        "response_id":          response_id,
        "target_found":         False,
        "target_bbox":          None,
        "text_content":         None,
        "text_readable":        None,
        "text_theme":           None,
        "cap_height_px":        None,
        "background_luminance": None,
        "text_luminance":       None,
        "text_color_rgb":       None,
        "background_color_rgb": None,
        "scene_description":    None,
        "target_confidence":    None,
        "stage3_flags":         [],
    }

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # ── Find target text region ───────────────────────────────────────────────
    target_bbox = None
    confidence = "low"
    theme = "other"

    # 1. Try open-vocabulary detection for common text carriers
    for query, t in [
        ("sign with text", "signage"),
        ("text on wall", "signage"),
        ("food label", "packaging"),
        ("product packaging", "packaging"),
        ("menu board", "signage"),
        ("digital screen with text", "digital_screen"),
        ("printed text", "print_media"),
        ("newspaper", "print_media"),
    ]:
        hits = detect_object(img, query)
        for h in hits:
            bx1, by1, bx2, by2 = h["bbox_frac"]
            # Must not overlap heavily with the reference card
            if card_bbox_frac:
                cx1, cy1, cx2, cy2 = card_bbox_frac
                overlap_x = max(0, min(bx2, cx2) - max(bx1, cx1))
                overlap_y = max(0, min(by2, cy2) - max(by1, cy1))
                overlap_area = overlap_x * overlap_y
                target_area = (bx2 - bx1) * (by2 - by1)
                if target_area > 0 and overlap_area / target_area > 0.5:
                    continue  # skip — mostly overlaps the card
            target_bbox = h["bbox_frac"]
            theme = t
            confidence = "medium"
            break
        if target_bbox:
            break

    # 2. Fallback: dense caption — pick largest non-card region with text label
    if target_bbox is None:
        regions = dense_caption(img)
        for r in regions:
            if not _is_text_region(r["label"]):
                continue
            bx1, by1, bx2, by2 = r["bbox_frac"]
            if card_bbox_frac:
                cx1, cy1, cx2, cy2 = card_bbox_frac
                overlap_x = max(0, min(bx2, cx2) - max(bx1, cx1))
                overlap_y = max(0, min(by2, cy2) - max(by1, cy1))
                overlap_area = overlap_x * overlap_y
                target_area = (bx2 - bx1) * (by2 - by1)
                if target_area > 0 and overlap_area / target_area > 0.5:
                    continue
            target_bbox = r["bbox_frac"]
            confidence = "low"
            break

    if target_bbox is None:
        result["stage3_flags"].append("target_not_found")
        return result

    result["target_found"]      = True
    result["target_bbox"]       = target_bbox
    result["target_confidence"] = confidence
    result["text_theme"]        = theme

    # ── OCR on full image, estimate cap height ────────────────────────────────
    ocr_regions = ocr_with_regions(img)
    result["cap_height_px"] = _estimate_cap_height(ocr_regions, target_bbox, (W, H))

    # Collect text content from OCR regions inside target bbox
    tx1, ty1, tx2, ty2 = target_bbox
    texts = [
        r["text"] for r in ocr_regions
        if (r["bbox_frac"][0] >= tx1 - 0.05 and r["bbox_frac"][2] <= tx2 + 0.05
            and r["bbox_frac"][1] >= ty1 - 0.05 and r["bbox_frac"][3] <= ty2 + 0.05)
    ]
    result["text_content"]  = " ".join(texts)[:300] if texts else None
    result["text_readable"] = bool(texts)

    # ── Color analysis from crop ──────────────────────────────────────────────
    target_crop = bbox_crop(img, target_bbox)
    bg_rgb   = _dominant_color(target_crop, region="border")
    text_rgb = _dominant_color(target_crop, region="all")

    result["background_color_rgb"] = bg_rgb
    result["text_color_rgb"]       = text_rgb

    bg_lum   = 0.2126*bg_rgb[0] + 0.7152*bg_rgb[1] + 0.0722*bg_rgb[2]
    text_lum = 0.2126*text_rgb[0] + 0.7152*text_rgb[1] + 0.0722*text_rgb[2]
    result["background_luminance"] = "light" if bg_lum > 127 else "dark"
    result["text_luminance"]       = "light" if text_lum > 127 else "dark"

    # ── Scene description ─────────────────────────────────────────────────────
    result["scene_description"] = caption(img)

    if confidence == "low":
        result["stage3_flags"].append("low_confidence")
    if not result["text_readable"]:
        result["stage3_flags"].append("text_unreadable")

    # ── Save target crop ──────────────────────────────────────────────────────
    crops_dir.mkdir(parents=True, exist_ok=True)
    try:
        img_np = np.array(img)
        sam_mask = refine_mask_with_sam(img_np, target_bbox)
        if sam_mask is not None:
            masked = img_np.copy()
            masked[~sam_mask] = 255
            target_img = Image.fromarray(masked)
        else:
            target_img = target_crop
        target_img.save(crops_dir / f"{response_id}_target.jpg")
    except Exception as e:
        log.warning(f"{response_id}: crop save failed — {e}")
        result["stage3_flags"].append("crop_save_failed")

    return result