"""
Stage 2 — Reference Card Detection & Calibration
================================================
Updates:
- Fixed bbox_crop height calculation bug (y2).
- Updated _ocr_anchor_gate to use check_ocr_anchors with fuzzy logic and len >= 1.
- Keeps dictionary clean of raw PIL objects for Parquet compatibility.
"""

import math
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import difflib

from config import (
    CALIB_BAR_CM, MAX_CARD_ANGLE_DEG,
    MIN_PX_PER_CM, MAX_PX_PER_CM, MIN_BAR_WIDTH_PX,
)
from local_model import detect_object, ocr_with_regions

log = logging.getLogger(__name__)

_CARD_ANCHORS = [
    "wicked", "words", "visual", "reading", "acuity", 
    "this bar", "bar", "8 cm", "8cm", "challenges", "cm"
]

def check_ocr_anchors(ocr_text: str) -> bool:
    """
    Relaxed anchor matching to account for OCR noise, blur, or severe angles.
    Checks substring containment across the whole text string first,
    then falls back to token-level fuzzy matching (>65% similarity).
    """
    text_lower = ocr_text.lower()

    # 1. Direct fast substring check (catches "8cm", "bar", "wicked", etc.)
    if any(anchor in text_lower for anchor in _CARD_ANCHORS):
        return True

    # 2. Relaxed fuzzy token matching (>65% similarity)
    words = text_lower.split()
    for anchor in _CARD_ANCHORS:
        if any(difflib.SequenceMatcher(None, anchor, w).ratio() > 0.65 for w in words):
            return True

    return False


# ── Relaxed Bar detection parameters ──────────────────────────────────────────
_BAR_LUMINANCE_THRESH = 130   # Raised: accepts shadowed/unevenly lit black bars
_BAR_MIN_ASPECT       = 2.0   # Lowered: accepts angled or foreshortened views
_BAR_MIN_WIDTH_FRAC   = 0.15  # Accepts smaller card bounding boxes

# ── Helpers ───────────────────────────────────────────────────────────────────

def bbox_crop(img: Image.Image, bbox_frac: list[float]) -> Image.Image:
    """Correctly crops PIL image using fractional bounding box [x1, y1, x2, y2]."""
    W, H = img.size
    x1, y1, x2, y2 = bbox_frac
    return img.crop((
        max(0, int(x1 * W)), max(0, int(y1 * H)),
        min(W, int(x2 * W)), min(H, int(y2 * H)), # ✅ Fixed y2 calculation
    ))


# ── OCR anchor gate ───────────────────────────────────────────────────────────

def _ocr_anchor_gate(img: Image.Image) -> dict:
    """
    Run OCR on the full image and check for Wicked Words card vocabulary.
    Returns {pass: bool, matched: list, ocr_regions: list}.
    """
    try:
        ocr_regions = ocr_with_regions(img)
    except Exception as e:
        log.warning(f"OCR failed: {e}")
        return {"pass": False, "matched": [], "ocr_regions": []}

    full_text = " ".join(r["text"].lower() for r in ocr_regions)
    matched = [a for a in _CARD_ANCHORS if a in full_text]
    
    # ✅ FIX: Use check_ocr_anchors() so fuzzy matching & single-token hits pass
    passed = check_ocr_anchors(full_text)

    log.debug(f"OCR gate: passed={passed}, matched direct={matched}")
    return {"pass": passed, "matched": matched, "ocr_regions": ocr_regions}


# ── Bar detection (full-width top/bottom strips) ───────────────────────────────

def _find_bar_in_strip(card_np: np.ndarray, strip: str) -> dict | None:
    """Search for the calibration bar in a horizontal strip of the card."""
    H, W = card_np.shape[:2]
    strip_h = max(4, int(H * 0.20))

    if strip == "top":
        region = card_np[:strip_h, :]
        y_offset = 0
    else:
        region = card_np[H - strip_h:, :]
        y_offset = H - strip_h

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    # Find the single darkest row — that is the bar centre
    row_means = gray.mean(axis=1)
    bar_row = int(np.argmin(row_means))

    # Expand outward from bar_row while rows stay dark
    threshold = min(row_means[bar_row] * 3.0, _BAR_LUMINANCE_THRESH)
    top_row = bar_row
    while top_row > 0 and row_means[top_row - 1] < threshold:
        top_row -= 1
    bot_row = bar_row
    while bot_row < len(row_means) - 1 and row_means[bot_row + 1] < threshold:
        bot_row += 1

    bar_h = bot_row - top_row + 1

    # Find horizontal extent: columns where the bar row is dark
    bar_slice = gray[top_row:bot_row + 1, :]
    col_means = bar_slice.mean(axis=0)
    dark_cols = np.where(col_means < threshold)[0]
    if len(dark_cols) < _BAR_MIN_WIDTH_FRAC * W:
        return None

    x_start, x_end = int(dark_cols[0]), int(dark_cols[-1])
    bar_w = x_end - x_start + 1
    aspect = bar_w / max(bar_h, 1)

    if aspect < _BAR_MIN_ASPECT:
        return None

    return {
        "bar_width_px":  float(bar_w),
        "bar_height_px": float(bar_h),
        "local_x1": x_start,
        "local_y1": top_row + y_offset,
        "local_x2": x_end,
        "local_y2": bot_row + y_offset,
        "strip": strip,
        "aspect": round(aspect, 2),
    }


def _detect_calibration_bar(
    card_img: Image.Image,
    card_bbox_frac: list[float],
    full_img_size: tuple,
) -> dict:
    card_np = np.array(card_img.convert("RGB"))
    card_H, card_W = card_np.shape[:2]
    cx1, cy1, cx2, cy2 = card_bbox_frac

    for strip in ["top", "bottom"]:
        bar = _find_bar_in_strip(card_np, strip)
        if bar is None:
            continue

        def to_full_frac_x(lx): return cx1 + (lx / card_W) * (cx2 - cx1)
        def to_full_frac_y(ly): return cy1 + (ly / card_H) * (cy2 - cy1)

        bar_frac = [
            max(0.0, to_full_frac_x(bar["local_x1"])),
            max(0.0, to_full_frac_y(bar["local_y1"])),
            min(1.0, to_full_frac_x(bar["local_x2"])),
            min(1.0, to_full_frac_y(bar["local_y2"])),
        ]
        log.info(
            f"Bar found in {strip} strip: "
            f"{bar['bar_width_px']:.0f}×{bar['bar_height_px']:.0f}px "
            f"aspect={bar['aspect']}"
        )
        return {"found": True, "bbox_frac": bar_frac,
                "bar_width_px": bar["bar_width_px"], "strip": strip}

    return {"found": False, "bbox_frac": None, "bar_width_px": None}


# ── Card detection with specific queries and confidence estimation ───────────

def _detect_card(image: Image.Image) -> dict:
    """Locate the reference card using object detection queries."""
    specific_queries = [
        "wicked words card",
        "visual acuity chart card",
        "reference card with eye chart",
        "postcard with text and eye test",
    ]
    generic_queries = [
        "white card held in hand",
        "white rectangular card",
        "card",
    ]

    for query in specific_queries + generic_queries:
        try:
            detections = detect_object(image, query)
        except Exception:
            continue

        for d in detections:
            x1, y1, x2, y2 = d["bbox_frac"]
            area_frac = (x2 - x1) * (y2 - y1)
            width = x2 - x1
            height = y2 - y1
            aspect = width / height if height > 0 else 0

            if area_frac > 0.01 and 0.5 <= aspect <= 3.0: # Relaxed aspect/area cutoffs
                confidence = round(d.get("score", 0.95), 2)
                log.info(f"Card detected via '{query}': area={area_frac:.1%} aspect={aspect:.2f} score={confidence}")
                return {
                    "found": True,
                    "bbox_frac": d["bbox_frac"],
                    "confidence": confidence
                }

    return {"found": False, "bbox_frac": None, "confidence": None}


# ── Scale factor ──────────────────────────────────────────────────────────────

def compute_scale_factor(
    img_width: int,
    img_height: int,
    bar_bbox_frac: list[float],
    card_angle_deg: float = 0.0,
) -> dict:
    x1, y1, x2, y2 = bar_bbox_frac
    apparent_bar_px = (x2 - x1) * img_width

    if apparent_bar_px < MIN_BAR_WIDTH_PX:
        return {"px_per_cm": None,
                "error": f"bar too narrow ({apparent_bar_px:.1f}px < {MIN_BAR_WIDTH_PX})"}

    tilt_corrected = abs(card_angle_deg) > 5
    if tilt_corrected:
        angle_rad = math.radians(min(abs(card_angle_deg), 60))
        corrected_px = apparent_bar_px / math.cos(angle_rad)
    else:
        corrected_px = apparent_bar_px

    px_per_cm = corrected_px / CALIB_BAR_CM

    if not (MIN_PX_PER_CM <= px_per_cm <= MAX_PX_PER_CM):
        return {
            "px_per_cm": round(px_per_cm, 2),
            "tilt_corrected": tilt_corrected,
            "error": f"px_per_cm={px_per_cm:.1f} outside [{MIN_PX_PER_CM}, {MAX_PX_PER_CM}]",
        }

    return {
        "px_per_cm": round(px_per_cm, 2),
        "apparent_bar_px": round(apparent_bar_px, 1),
        "corrected_bar_px": round(corrected_px, 1),
        "tilt_corrected": tilt_corrected,
        "error": None,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def process_image(response_id: str, image_path: Path, crops_dir: Path) -> dict:
    result = {
        "response_id":          response_id,
        "card_found":           False,
        "card_confidence":      None,
        "px_per_cm":            None,
        "card_bbox":            None,
        "calibration_bar_bbox": None,
        "card_angle_deg":       0.0,
        "scale_error":          None,
        "stage2_flags":         [],
        "ocr_anchor_matches":   [],
        "bar_width_px":        None,
    }

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.error(f"{response_id}: Failed to open image — {e}")
        result["stage2_flags"].append("image_open_failed")
        return result

    W, H = img.size

    # ── OCR gate ───────────────────────────────────────────────────────────────
    gate = _ocr_anchor_gate(img)
    result["ocr_anchor_matches"] = gate["matched"]

    if not gate["pass"]:
        result["card_found"] = False
        result["stage2_flags"].append("card_absent_ocr_gate")
        log.info(f"{response_id}: OCR gate REJECT ({len(gate['matched'])} anchors). Card absent.")
        return result

    # ── Card detection (OVD) ──────────────────────────────────────────────────
    card = _detect_card(img)
    result["card_found"]      = card["found"]
    result["card_bbox"]       = card["bbox_frac"]
    result["card_confidence"] = card["confidence"]

    if not card["found"]:
        result["stage2_flags"].append("card_not_found_ovd")
        log.info(f"{response_id}: Card not found by OVD despite OCR anchors present.")
        return result

    # ── Bar detection ─────────────────────────────────────────────────────────
    card_img = bbox_crop(img, card["bbox_frac"])
    bar = _detect_calibration_bar(card_img, card["bbox_frac"], (W, H))
    result["calibration_bar_bbox"] = bar.get("bbox_frac")
    result["bar_width_px"] = bar.get("bar_width_px")    
    if not bar["found"]:
        result["scale_error"] = "bar_not_found"
        result["stage2_flags"].append("bar_not_found")
        log.info(f"{response_id}: Card found but calibration bar not detected.")
        return result

    # ── Scale factor ───────────────────────────────────────────────────────────
    scale = compute_scale_factor(W, H, bar["bbox_frac"], result["card_angle_deg"])
    result["px_per_cm"]   = scale.get("px_per_cm")
    result["scale_error"] = scale.get("error")
    if scale.get("error"):
        result["stage2_flags"].append("scale_out_of_range")

    

    log.info(
        f"{response_id}: card_found=True confidence={result['card_confidence']} "
        f"px_per_cm={result['px_per_cm']} bar_strip={bar.get('strip')}"
    )
    return result