"""
Stage 2 — Reference Card Detection & Calibration  (v3 — fixed)
===============================================================
Key fixes over v2
-----------------
FIX 1  OCR hard gate runs BEFORE any Florence OVD call.
       If zero Wicked Words anchor strings are found in the image,
       card_found=False immediately — no OVD called, no false positive.
       This kills the p-004/p-005 "dddd" false positives at zero cost.

FIX 2  Bar detection searches TOP strip and BOTTOM strip of the card
       (full-width horizontal bands), not corner quadrants.
       The calibration bar spans the full card width at top AND bottom.

FIX 3  Returns masked_img — the full image with the card region
       painted out (median background fill). Stage 3 receives this
       instead of the original, so Florence never sees the card text
       when searching for the target text.

FIX 4  OVD query order is now specific → generic (was generic first,
       causing "paper" to match random regions before the real card
       was even tried with a meaningful query).
"""

import math
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

from config import (
    CALIB_BAR_CM, MAX_CARD_ANGLE_DEG,
    MIN_PX_PER_CM, MAX_PX_PER_CM, MIN_BAR_WIDTH_PX,
)
from local_model import detect_object, ocr_with_regions

log = logging.getLogger(__name__)

# ── OCR anchor strings — must find MIN_ANCHORS before OVD is attempted ────────
_CARD_ANCHORS = [
    "wicked", "words", "logmar", "snellen",
    "8 cm", "20/20", "20/", "visual acuity", "this bar", "reading",
]
_MIN_ANCHORS = 2   # at least 2 must appear

# ── Bar detection parameters ──────────────────────────────────────────────────
_BAR_LUMINANCE_THRESH = 80    # pixels darker than this are "black bar" candidates
_BAR_MIN_ASPECT       = 5.0   # bar width / height must exceed this
_BAR_MIN_WIDTH_FRAC   = 0.40  # bar must span ≥ 40% of card width


# ── Helpers ───────────────────────────────────────────────────────────────────

def bbox_crop(img: Image.Image, bbox_frac: list[float]) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = bbox_frac
    return img.crop((
        max(0, int(x1 * W)), max(0, int(y1 * H)),
        min(W, int(x2 * W)), min(H, int(y2 * H)),
    ))


def mask_card_from_image(img: Image.Image, card_bbox_frac: list[float]) -> Image.Image:
    """
    Return a copy of img with the card region painted out.
    Fill colour = median of the surrounding border pixels (blends naturally).
    Stage 3 uses this so Florence never sees the Snellen chart / card text
    when hunting for the participant's target text.
    """
    img_np = np.array(img.convert("RGB"))
    H, W = img_np.shape[:2]
    x1, y1, x2, y2 = card_bbox_frac
    px1, py1 = max(0, int(x1 * W)), max(0, int(y1 * H))
    px2, py2 = min(W, int(x2 * W)), min(H, int(y2 * H))

    # Sample border pixels around the card region for fill colour
    border_pixels = []
    pad = 10
    for row in [max(0, py1 - pad), min(H - 1, py2 + pad)]:
        border_pixels.append(img_np[row, px1:px2].reshape(-1, 3))
    for col in [max(0, px1 - pad), min(W - 1, px2 + pad)]:
        border_pixels.append(img_np[py1:py2, col].reshape(-1, 3))

    if border_pixels:
        fill = np.median(np.vstack(border_pixels), axis=0).astype(np.uint8)
    else:
        fill = np.array([200, 200, 200], dtype=np.uint8)

    masked = img_np.copy()
    masked[py1:py2, px1:px2] = fill
    return Image.fromarray(masked)


# ── FIX 1: OCR anchor gate ────────────────────────────────────────────────────

def _ocr_anchor_gate(img: Image.Image) -> dict:
    """
    Run OCR on the full image and check for Wicked Words card vocabulary.
    Returns {pass: bool, matched: list, ocr_regions: list}.
    If pass=False, card is definitively absent — skip all OVD calls.
    """
    try:
        ocr_regions = ocr_with_regions(img)
    except Exception as e:
        log.warning(f"OCR failed: {e}")
        return {"pass": False, "matched": [], "ocr_regions": []}

    full_text = " ".join(r["text"].lower() for r in ocr_regions)
    matched = [a for a in _CARD_ANCHORS if a in full_text]
    passed = len(matched) >= _MIN_ANCHORS

    log.debug(f"OCR gate: {len(matched)} anchors matched ({matched})")
    return {"pass": passed, "matched": matched, "ocr_regions": ocr_regions}


# ── FIX 2: Correct bar detection (full-width top/bottom strips) ───────────────

def _find_bar_in_strip(card_np: np.ndarray, strip: str) -> dict | None:
    """
    Search for the calibration bar in a horizontal strip of the card.

    'top'    → top 20% of card height (bar sits at the very top)
    'bottom' → bottom 20% of card height (bar sits at the very bottom)

    The bar must:
      - be very dark (luminance < _BAR_LUMINANCE_THRESH)
      - have aspect ratio > _BAR_MIN_ASPECT  (much wider than tall)
      - span ≥ _BAR_MIN_WIDTH_FRAC of card width

    Returns local-coordinate bbox dict or None.
    """
    H, W = card_np.shape[:2]
    strip_h = max(4, int(H * 0.20))

    if strip == "top":
        region = card_np[:strip_h, :]
        y_offset = 0
    else:   # bottom
        region = card_np[H - strip_h:, :]
        y_offset = H - strip_h

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, _BAR_LUMINANCE_THRESH, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 2 or w < _BAR_MIN_WIDTH_FRAC * W:
            continue
        aspect = w / h
        if aspect < _BAR_MIN_ASPECT:
            continue
        score = aspect * (w / W)   # wider + more spanning = better
        if score > best_score:
            best_score = score
            best = {
                "bar_width_px": float(w),
                "bar_height_px": float(h),
                "local_x1": x,
                "local_y1": y + y_offset,
                "local_x2": x + w,
                "local_y2": y + y_offset + h,
                "strip": strip,
                "aspect": round(aspect, 2),
            }
    return best


def _detect_calibration_bar(
    card_img: Image.Image,
    card_bbox_frac: list[float],
    full_img_size: tuple,
) -> dict:
    """
    Find the 8 cm bar. Tries top strip first, then bottom.
    Maps local card coordinates back to full-image fractional coordinates.
    """
    card_np = np.array(card_img.convert("RGB"))
    card_H, card_W = card_np.shape[:2]
    full_W, full_H = full_img_size
    cx1, cy1, cx2, cy2 = card_bbox_frac

    for strip in ["top", "bottom"]:
        bar = _find_bar_in_strip(card_np, strip)
        if bar is None:
            continue

        # Local card px → full image fractional
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


# ── FIX 4: Card detection — specific queries first ────────────────────────────

def _detect_card(image: Image.Image) -> dict:
    """
    Locate the Wicked Words reference card.
    Queries go most-specific → least-specific.
    Only falls back to generic terms if specific ones fail.
    Area gate (> 2% of image) prevents tiny spurious detections.
    """
    # Specific queries first — these are unlikely to match random objects
    specific_queries = [
        "wicked words card",
        "visual acuity chart card",
        "reference card with eye chart",
        "postcard with text and eye test",
    ]
    # Generic fallbacks — only tried if all specific queries fail
    generic_queries = [
        "white card held in hand",
        "white rectangular card",
    ]

    for query in specific_queries + generic_queries:
        try:
            detections = detect_object(image, query)
        except Exception:
            continue

        for d in detections:
            x1, y1, x2, y2 = d["bbox_frac"]
            area_frac = (x2 - x1) * (y2 - y1)
            # Card aspect ratio check: must be roughly landscape (0.8–2.5)
            width = x2 - x1
            height = y2 - y1
            aspect = width / height if height > 0 else 0
            if area_frac > 0.02 and 0.8 <= aspect <= 2.5:
                log.info(f"Card detected via '{query}': area={area_frac:.1%} aspect={aspect:.2f}")
                return {"found": True, "bbox_frac": d["bbox_frac"]}

    return {"found": False, "bbox_frac": None}


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
    """
    Returns result dict AND masked_img (card painted out, for Stage 3).
    Caller should use result["masked_img"] in stage3.process_image().
    """
    result = {
        "response_id":        response_id,
        "card_found":         False,
        "px_per_cm":          None,
        "card_bbox":          None,
        "calibration_bar_bbox": None,
        "card_angle_deg":     0.0,
        "scale_error":        None,
        "stage2_flags":       [],
        "masked_img":         None,   # PIL Image with card painted out
        "ocr_anchor_matches": [],
    }

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # ── FIX 1: OCR gate — reject card-absent images instantly ─────────────────
    gate = _ocr_anchor_gate(img)
    result["ocr_anchor_matches"] = gate["matched"]

    if not gate["pass"]:
        result["card_found"] = False
        result["stage2_flags"].append("card_absent_ocr_gate")
        log.info(
            f"{response_id}: OCR gate REJECT "
            f"({len(gate['matched'])}/{_MIN_ANCHORS} anchors). "
            f"Card definitively absent."
        )
        # Still return original image as masked_img so Stage 3 can run
        result["masked_img"] = img
        return result

    # ── Card detection (OVD) ──────────────────────────────────────────────────
    card = _detect_card(img)
    result["card_found"] = card["found"]
    result["card_bbox"]  = card["bbox_frac"]

    if not card["found"]:
        result["stage2_flags"].append("card_not_found_ovd")
        result["masked_img"] = img
        log.info(f"{response_id}: card not found by OVD despite OCR anchors present")
        return result

    # ── FIX 3: Mask card from image for Stage 3 ───────────────────────────────
    masked_img = mask_card_from_image(img, card["bbox_frac"])
    result["masked_img"] = masked_img

    # ── FIX 2: Bar detection in correct strips ────────────────────────────────
    card_img = bbox_crop(img, card["bbox_frac"])
    bar = _detect_calibration_bar(card_img, card["bbox_frac"], (W, H))
    result["calibration_bar_bbox"] = bar.get("bbox_frac")

    if not bar["found"]:
        result["scale_error"] = "bar_not_found"
        result["stage2_flags"].append("bar_not_found")
        log.info(f"{response_id}: card found but bar not detected")
        return result

    # ── Scale factor ───────────────────────────────────────────────────────────
    scale = compute_scale_factor(W, H, bar["bbox_frac"], result["card_angle_deg"])
    result["px_per_cm"]   = scale.get("px_per_cm")
    result["scale_error"] = scale.get("error")
    if scale.get("error"):
        result["stage2_flags"].append("scale_out_of_range")

    # ── Save card crop ─────────────────────────────────────────────────────────
    try:
        crops_dir.mkdir(parents=True, exist_ok=True)
        bbox_crop(img, card["bbox_frac"]).save(
            crops_dir / f"{response_id}_card.jpg"
        )
    except Exception as e:
        log.warning(f"{response_id}: card crop save failed — {e}")

    log.info(
        f"{response_id}: card_found=True "
        f"px_per_cm={result['px_per_cm']} "
        f"bar_strip={bar.get('strip')} "
        f"anchors={gate['matched']}"
    )
    return result