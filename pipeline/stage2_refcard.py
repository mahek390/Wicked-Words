"""
Stage 2 — Reference Card Detection & Calibration (Fixed)
==================================================
Uses Florence-2 + OpenCV contour analysis to:
  1. Detect the Wicked Words postcard bounding box
  2. Locate top-right or bottom-left 8 cm calibration bars
  3. Compute accurate px/cm scale factor
"""

import math
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from config import (
    CALIB_BAR_CM, MAX_CARD_ANGLE_DEG, MIN_PX_PER_CM, MAX_PX_PER_CM,
    MIN_BAR_WIDTH_PX,
)
from local_model import detect_object, dense_caption, ocr_with_regions

log = logging.getLogger(__name__)


def bbox_crop(img: Image.Image, bbox_frac: list[float]) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = bbox_frac
    return img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))


def _detect_card(image: Image.Image) -> dict:
    for query in ["postcard", "reference card", "white card with text", "paper"]:
        detections = detect_object(image, query)
        if detections:
            d = detections[0]
            x1, y1, x2, y2 = d["bbox_frac"]
            area_frac = (x2 - x1) * (y2 - y1)
            if area_frac > 0.02:
                log.info(f"Card found via query '{query}', area={area_frac:.2%}")
                return {"found": True, "bbox_frac": d["bbox_frac"]}

    regions = dense_caption(image)
    for r in regions:
        lbl = r["label"].lower()
        if any(k in lbl for k in ["card", "postcard", "sign", "paper", "note"]):
            x1, y1, x2, y2 = r["bbox_frac"]
            area_frac = (x2 - x1) * (y2 - y1)
            if area_frac > 0.02:
                log.info(f"Card found via dense caption '{r['label']}'")
                return {"found": True, "bbox_frac": r["bbox_frac"]}

    return {"found": False, "bbox_frac": None}


def _find_bar_in_quadrant(card_np: np.ndarray, quad_type: str) -> dict:
    """
    Finds a horizontal black calibration bar in specific card quadrants:
    - 'top_right': Top 35% height, Right 50% width
    - 'bottom_left': Bottom 35% height, Left 50% width
    """
    H, W, _ = card_np.shape

    if quad_type == "top_right":
        ymin, ymax = 0, int(H * 0.35)
        xmin, xmax = int(W * 0.50), W
    else:  # bottom_left
        ymin, ymax = int(H * 0.65), H
        xmin, xmax = 0, int(W * 0.50)

    crop = card_np[ymin:ymax, xmin:xmax]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Threshold for dark regions (black bar)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bar = None
    max_width = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0

        # Calibration bar properties: horizontal aspect ratio (> 3.0), non-trivial size
        if aspect_ratio >= 3.0 and w > 20 and w > max_width:
            max_width = w
            # Local coordinates inside card
            local_x1 = xmin + x
            local_y1 = ymin + y
            local_x2 = local_x1 + w
            local_y2 = local_y1 + h

            best_bar = {
                "bar_width_px": float(w),
                "card_bbox_frac": [
                    local_x1 / W,
                    local_y1 / H,
                    local_x2 / W,
                    local_y2 / H
                ]
            }

    return best_bar


def _detect_calibration_bar(card_img: Image.Image, card_bbox_frac: list[float], full_img: Image.Image) -> dict:
    """
    Locates 8 cm calibration bar using OpenCV sub-quadrant scanning.
    """
    card_np = np.array(card_img)
    cx1, cy1, cx2, cy2 = card_bbox_frac

    # Try top-right quadrant first, then bottom-left
    for quad in ["top_right", "bottom_left"]:
        bar_data = _find_bar_in_quadrant(card_np, quad)
        if bar_data:
            bx1, by1, bx2, by2 = bar_data["card_bbox_frac"]
            
            # Map back to full image normalized fractions
            bar_frac_full = [
                max(0.0, cx1 + bx1 * (cx2 - cx1)),
                max(0.0, cy1 + by1 * (cy2 - cy1)),
                min(1.0, cx1 + bx2 * (cx2 - cx1)),
                min(1.0, cy1 + by2 * (cy2 - cy1)),
            ]
            
            log.info(f"Calibration bar detected in {quad} quadrant: width={bar_data['bar_width_px']:.1f}px")
            return {
                "found": True,
                "bbox_frac": bar_frac_full,
                "bar_width_px": bar_data["bar_width_px"],
                "method": f"contour_{quad}"
            }

    return {"found": False, "bbox_frac": None, "bar_width_px": None}


def compute_scale_factor(img_width: int, img_height: int, bar_bbox_frac: list[float], card_angle_deg: float = 0.0) -> dict:
    x1, y1, x2, y2 = bar_bbox_frac
    apparent_bar_px = (x2 - x1) * img_width

    if apparent_bar_px < MIN_BAR_WIDTH_PX:
        return {"px_per_cm": None, "error": f"bar too narrow ({apparent_bar_px:.1f}px)"}

    angle_rad = math.radians(min(abs(card_angle_deg), 60))
    tilt_corrected = abs(card_angle_deg) > 5
    corrected_bar_px = apparent_bar_px / math.cos(angle_rad) if tilt_corrected else apparent_bar_px
    px_per_cm = corrected_bar_px / CALIB_BAR_CM

    if not (MIN_PX_PER_CM <= px_per_cm <= MAX_PX_PER_CM):
        return {
            "px_per_cm": round(px_per_cm, 2),
            "tilt_corrected": tilt_corrected,
            "error": f"px_per_cm={px_per_cm:.1f} outside plausible range"
        }

    return {
        "px_per_cm": round(px_per_cm, 2),
        "apparent_bar_px": round(apparent_bar_px, 1),
        "corrected_bar_px": round(corrected_bar_px, 1),
        "tilt_corrected": tilt_corrected,
        "error": None,
    }


def process_image(response_id: str, image_path: Path, crops_dir: Path) -> dict:
    result = {
        "response_id": response_id,
        "card_found": False,
        "px_per_cm": None,
        "card_bbox": None,
        "calibration_bar_bbox": None,
        "card_angle_deg": 0.0,
        "scale_error": None,
        "stage2_flags": [],
    }

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Detect Card
    card = _detect_card(img)
    result["card_found"] = card["found"]
    result["card_bbox"] = card["bbox_frac"]

    if not card["found"]:
        result["stage2_flags"].append("card_not_found")
        return result

    # Crop Card & Find Bar
    card_img = bbox_crop(img, card["bbox_frac"])
    bar = _detect_calibration_bar(card_img, card["bbox_frac"], img)
    result["calibration_bar_bbox"] = bar.get("bbox_frac")

    if not bar["found"]:
        result["scale_error"] = "bar_not_found"
        result["stage2_flags"].append("bar_not_found")
        return result

    # Compute Scale
    scale = compute_scale_factor(W, H, bar["bbox_frac"], result["card_angle_deg"])
    result["px_per_cm"] = scale.get("px_per_cm")
    result["scale_error"] = scale.get("error")
    if scale.get("error"):
        result["stage2_flags"].append("scale_error")

    # Save Card Crop
    crops_dir.mkdir(parents=True, exist_ok=True)
    card_img.save(crops_dir / f"{response_id}_card.jpg")

    return result