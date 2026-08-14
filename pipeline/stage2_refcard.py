"""
Stage 2 — Reference Card Detection & Calibration
================================================
Updates:
- OCR Spatial Search: Locates '8 cm' / 'this bar' text anchor to lock onto the calibration bar.
- Orientation Invariance: Rotates card in 90-degree steps to handle rotated photos.
- Parquet Compatible: Clean return dictionary with no raw PIL image objects.
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

_BAR_MIN_ASPECT = 2.0
_BAR_MIN_WIDTH_FRAC = 0.15
_BAR_MAX_HEIGHT_FRAC = 0.15
_BAR_LUMINANCE_THRESH = 100


def check_ocr_anchors(ocr_text: str) -> bool:
    """Checks for presence of key card vocabulary using substring and fuzzy token matching."""
    text_lower = ocr_text.lower()
    if any(anchor in text_lower for anchor in _CARD_ANCHORS):
        return True

    words = text_lower.split()
    for anchor in _CARD_ANCHORS:
        if any(difflib.SequenceMatcher(None, anchor, w).ratio() > 0.65 for w in words):
            return True

    return False


def bbox_crop(img: Image.Image, bbox_frac: list[float]) -> Image.Image:
    """Crops PIL image using fractional bounding box [x1, y1, x2, y2]."""
    W, H = img.size
    x1, y1, x2, y2 = bbox_frac
    return img.crop((
        max(0, int(x1 * W)), max(0, int(y1 * H)),
        min(W, int(x2 * W)), min(H, int(y2 * H)),
    ))


def _ocr_anchor_gate(img: Image.Image) -> dict:
    """Run OCR on full image to verify presence of reference card anchors."""
    try:
        ocr_regions = ocr_with_regions(img)
    except Exception as e:
        log.warning(f"OCR failed: {e}")
        return {"pass": False, "matched": [], "ocr_regions": []}

    full_text = " ".join(r["text"].lower() for r in ocr_regions)
    matched = [a for a in _CARD_ANCHORS if a in full_text]
    passed = check_ocr_anchors(full_text)

    return {"pass": passed, "matched": matched, "ocr_regions": ocr_regions}


# ── OCR-Guided Bar Localizer ─────────────────────────────────────────────────

def _find_bar_near_ocr_anchor(card_np: np.ndarray, ocr_regions: list) -> dict | None:
    """
    Looks specifically near OCR detections containing '8 cm' or 'this bar'.
    This guarantees we isolate the calibration bar from other dark lines.
    """
    H, W = card_np.shape[:2]
    anchor_box = None

    for r in ocr_regions:
        txt = r.get("text", "").lower()
        if "8 cm" in txt or "8cm" in txt or "this bar" in txt:
            # Found spatial anchor! Convert bounding box coordinates
            if "bbox_px" in r:
                anchor_box = r["bbox_px"]
            elif "bbox_frac" in r:
                ax1, ay1, ax2, ay2 = r["bbox_frac"]
                anchor_box = (int(ax1 * W), int(ay1 * H), int(ax2 * W), int(ay2 * H))
            break

    if not anchor_box:
        return None

    # Search in a region expanded around the text label (above/below/adjacent)
    ax1, ay1, ax2, ay2 = anchor_box
    search_y1 = max(0, ay1 - int(0.15 * H))
    search_y2 = min(H, ay2 + int(0.15 * H))
    search_x1 = max(0, ax1 - int(0.10 * W))
    search_x2 = min(W, ax2 + int(0.10 * W))

    roi = card_np[search_y1:search_y2, search_x1:search_x2]
    if roi.size == 0:
        return None

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray_roi, _BAR_LUMINANCE_THRESH, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_candidate = None
    max_score = 0.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / max(short_side, 1)

        if aspect >= _BAR_MIN_ASPECT and long_side >= (_BAR_MIN_WIDTH_FRAC * W * 0.5):
            area = cv2.contourArea(cnt)
            rect_area = w * h
            solidity = area / float(rect_area) if rect_area > 0 else 0
            
            if solidity > 0.5:
                score = aspect * solidity
                if score > max_score:
                    max_score = score
                    best_candidate = {
                        "bar_width_px": float(long_side),
                        "bar_height_px": float(short_side),
                        "local_x1": int(search_x1 + x),
                        "local_y1": int(search_y1 + y),
                        "local_x2": int(search_x1 + x + w),
                        "local_y2": int(search_y1 + y + h),
                        "aspect": round(aspect, 2),
                        "orientation": "horizontal" if w >= h else "vertical",
                        "method": "ocr_anchored"
                    }

    return best_candidate


# ── Contour-Based Multi-Orientation Fallback Detector ─────────────────────────

def _find_bar_oriented(card_np: np.ndarray) -> dict | None:
    """Fallback detector scanning entire card using adaptive thresholding & shape analysis."""
    H, W = card_np.shape[:2]
    gray = cv2.cvtColor(card_np, cv2.COLOR_RGB2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, _BAR_LUMINANCE_THRESH, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_candidate = None
    max_score = 0.0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        long_side = max(w, h)
        short_side = min(w, h)
        aspect = long_side / max(short_side, 1)

        if aspect >= _BAR_MIN_ASPECT and (long_side >= _BAR_MIN_WIDTH_FRAC * max(W, H)):
            if short_side <= _BAR_MAX_HEIGHT_FRAC * min(W, H):
                area = cv2.contourArea(cnt)
                rect_area = w * h
                solidity = area / float(rect_area) if rect_area > 0 else 0
                
                if solidity > 0.6:
                    score = aspect * solidity
                    if score > max_score:
                        max_score = score
                        best_candidate = {
                            "bar_width_px": float(long_side),
                            "bar_height_px": float(short_side),
                            "local_x1": int(x),
                            "local_y1": int(y),
                            "local_x2": int(x + w),
                            "local_y2": int(y + h),
                            "aspect": round(aspect, 2),
                            "orientation": "horizontal" if w >= h else "vertical",
                            "method": "contour_scan"
                        }

    return best_candidate


def _detect_calibration_bar(
    card_img: Image.Image,
    card_bbox_frac: list[float],
    full_img_size: tuple,
    ocr_regions: list = None
) -> dict:
    """
    Main calibration bar entry point. First attempts OCR-guided detection.
    If unanchored, tests cardinal rotations (0°, 90°, 180°, 270°) to find rotated bars.
    """
    orig_card_np = np.array(card_img.convert("RGB"))
    cx1, cy1, cx2, cy2 = card_bbox_frac

    def to_full_frac(lx, ly, current_w, current_h):
        fx = cx1 + (lx / current_w) * (cx2 - cx1)
        fy = cy1 + (ly / current_h) * (cy2 - cy1)
        return fx, fy

    # 1. Primary Strategy: Find bar spatially near "8 cm" / "this bar" text
    if ocr_regions:
        ocr_bar = _find_bar_near_ocr_anchor(orig_card_np, ocr_regions)
        if ocr_bar is not None:
            card_H, card_W = orig_card_np.shape[:2]
            fx1, fy1 = to_full_frac(ocr_bar["local_x1"], ocr_bar["local_y1"], card_W, card_H)
            fx2, fy2 = to_full_frac(ocr_bar["local_x2"], ocr_bar["local_y2"], card_W, card_H)

            bar_frac = [
                max(0.0, min(fx1, fx2)), max(0.0, min(fy1, fy2)),
                min(1.0, max(fx1, fx2)), min(1.0, max(fy1, fy2)),
            ]
            log.info(f"Calibration bar found via OCR Anchor: {ocr_bar['bar_width_px']:.0f}px")
            return {
                "found": True,
                "bbox_frac": bar_frac,
                "bar_width_px": ocr_bar["bar_width_px"],
                "orientation": ocr_bar["orientation"],
                "method": "ocr_anchored"
            }

    # 2. Fallback Strategy: Cardinal rotation scan using geometric shapes
    for rot_k in range(4):
        rotated_np = np.rot90(orig_card_np, k=rot_k)
        rH, rW = rotated_np.shape[:2]

        bar = _find_bar_oriented(rotated_np)
        if bar is not None:
            rx1, ry1, rx2, ry2 = bar["local_x1"], bar["local_y1"], bar["local_x2"], bar["local_y2"]

            # Coordinate re-projection math across rotations
            if rot_k == 0:
                x1, y1, x2, y2 = rx1, ry1, rx2, ry2
            elif rot_k == 1:
                x1, y1, x2, y2 = ry1, rW - rx2, ry2, rW - rx1
            elif rot_k == 2:
                x1, y1, x2, y2 = rW - rx2, rH - ry2, rW - rx1, rH - ry1
            elif rot_k == 3:
                x1, y1, x2, y2 = rH - ry2, rx1, rH - ry1, rx2

            card_H, card_W = orig_card_np.shape[:2]
            fx1, fy1 = to_full_frac(x1, y1, card_W, card_H)
            fx2, fy2 = to_full_frac(x2, y2, card_W, card_H)

            bar_frac = [
                max(0.0, min(fx1, fx2)), max(0.0, min(fy1, fy2)),
                min(1.0, max(fx1, fx2)), min(1.0, max(fy1, fy2)),
            ]

            log.info(f"Bar detected via shape scan (rot={rot_k*90}°): {bar['bar_width_px']:.0f}px")
            return {
                "found": True,
                "bbox_frac": bar_frac,
                "bar_width_px": bar["bar_width_px"],
                "orientation": bar["orientation"],
                "method": "contour_scan"
            }

    return {"found": False, "bbox_frac": None, "bar_width_px": None, "orientation": None}


# Specific queries carry higher base confidence than generic ones.
# Blended with detection area fraction to produce a real signal.
_CARD_QUERIES = [
    ("wicked words card",              0.90),
    ("visual acuity chart card",        0.85),
    ("reference card with eye chart",   0.80),
    ("postcard with text and eye test", 0.75),
    ("white card held in hand",         0.55),
    ("white rectangular card",          0.50),
    ("card",                            0.40),
]


def _detect_card(image: Image.Image) -> dict:
    """Locate the reference card using object detection queries."""
    for query, query_base in _CARD_QUERIES:
        try:
            detections = detect_object(image, query)
        except Exception:
            continue

        for d in detections:
            x1, y1, x2, y2 = d["bbox_frac"]
            area_frac = (x2 - x1) * (y2 - y1)
            aspect = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0

            if area_frac > 0.01 and 0.5 <= aspect <= 3.0:
                # 60% query specificity + 40% normalized detection area.
                # area_frac from detect_object is already normalized 0-1.
                confidence = round(0.6 * query_base + 0.4 * d.get("area_frac", area_frac), 3)
                log.info(f"Card detected via '{query}': area={area_frac:.1%} aspect={aspect:.2f} confidence={confidence}")
                return {
                    "found": True,
                    "bbox_frac": d["bbox_frac"],
                    "confidence": confidence,
                    "query": query,
                }

    return {"found": False, "bbox_frac": None, "confidence": None, "query": None}


def compute_scale_factor(
    apparent_bar_px: float | None,
    card_angle_deg: float = 0.0,
) -> dict:
    """Computes physical px_per_cm scale factor based on calibration bar width."""
    if apparent_bar_px is None or apparent_bar_px < MIN_BAR_WIDTH_PX:
        measured = f"{apparent_bar_px:.1f}" if apparent_bar_px is not None else "None"
        return {"px_per_cm": None, "error": f"bar too narrow ({measured}px < {MIN_BAR_WIDTH_PX})"}

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


def process_image(response_id: str, image_path: Path, crops_dir: Path) -> dict:
    """Stage 2 primary runner function."""
    result = {
        "response_id":          response_id,
        "card_found":           False,
        "card_confidence":      None,
        "px_per_cm":            None,
        "card_bbox":            None,
        "calibration_bar_bbox": None,
        "card_angle_deg":       0.0,
        "corrected_bar_px":     None,
        "scale_error":          None,
        "stage2_flags":         [],
        "ocr_anchor_matches":   [],
        "bar_width_px":         None,
        "calibration_bar_orientation": None,
        "card_detection_query": None,
        "privacy_flag":         False,
    }

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.error(f"{response_id}: Failed to open image — {e}")
        result["stage2_flags"].append("image_open_failed")
        return result

    W, H = img.size

    # 1. OCR Gate
    gate = _ocr_anchor_gate(img)
    result["ocr_anchor_matches"] = gate["matched"]

    if not gate["pass"]:
        result["stage2_flags"].append("card_absent_ocr_gate")
        log.info(f"{response_id}: OCR gate REJECT. Card absent.")
        return result

    # 2. Card Detection
    card = _detect_card(img)
    result["card_found"]           = card["found"]
    result["card_bbox"]             = card["bbox_frac"]
    result["card_confidence"]       = card["confidence"]
    result["card_detection_query"]  = card["query"]

    if not card["found"]:
        result["stage2_flags"].append("card_not_found_ovd")
        log.info(f"{response_id}: Card not found by OVD despite OCR anchors.")
        return result

    # 3. Bar Detection (Using OCR regions passed into detector)
    card_img = bbox_crop(img, card["bbox_frac"])
    bar = _detect_calibration_bar(card_img, card["bbox_frac"], (W, H), gate.get("ocr_regions"))
    
    result["calibration_bar_bbox"] = bar.get("bbox_frac")
    result["bar_width_px"] = bar.get("bar_width_px")
    result["calibration_bar_orientation"] = bar.get("orientation")

    if not bar["found"]:
        result["scale_error"] = "bar_not_found"
        result["stage2_flags"].append("bar_not_found")
        log.info(f"{response_id}: Card found but calibration bar not detected.")
        return result

    # 4. Scale Calculation
    scale = compute_scale_factor(bar.get("bar_width_px"), result["card_angle_deg"])
    result["px_per_cm"]       = scale.get("px_per_cm")
    result["corrected_bar_px"] = scale.get("corrected_bar_px")
    result["scale_error"]     = scale.get("error")

    if scale.get("error"):
        result["stage2_flags"].append("scale_out_of_range")

    log.info(
        f"{response_id}: card_found=True confidence={result['card_confidence']} "
        f"px_per_cm={result['px_per_cm']} method={bar.get('method')}"
    )
    return result