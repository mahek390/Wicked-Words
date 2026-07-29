"""
Stage 3 — Multi-Target Text Detection & Visual Metrics
======================================================
Calculates per text block:
- Bounding box coordinates (fractional & pixel)
- Absolute visual angle (degrees)
- Relative visual angle ratio (text height px / calibration bar px)
"""

import math
import logging
import numpy as np
from pathlib import Path
from PIL import Image

from config import CALIB_BAR_CM
from local_model import ocr_with_regions
from stage2_stage3_annotator import process_and_annotate_record

log = logging.getLogger(__name__)


def compute_visual_angle_deg(height_cm: float, distance_cm: float) -> float | None:
    """Calculates absolute visual angle in degrees: θ = 2 * arctan(h / (2 * d))"""
    if not height_cm or not distance_cm or distance_cm <= 0:
        return None
    angle_rad = 2 * math.atan(height_cm / (2 * distance_cm))
    return round(math.degrees(angle_rad), 3)


def process_image(
    response_id: str,
    image_path: Path,
    crops_dir: Path,
    px_per_cm: float | None = None,
    distance_cm: float | None = None,
    card_bbox_frac: list[float] | None = None,
) -> dict:
    result = {
        "response_id":   response_id,
        "target_found":  False,
        "text_blocks":   [],
        "stage3_flags":  [],
    }

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        log.error(f"{response_id}: Image open failed — {e}")
        result["stage3_flags"].append("image_open_failed")
        return result

    W, H = img.size

    try:
        ocr_regions = ocr_with_regions(img)
    except Exception as e:
        log.error(f"{response_id}: OCR failed — {e}")
        result["stage3_flags"].append("ocr_failed")
        return result

    # 1. Filter regions (only filter card overlap IF card_bbox_frac exists)
    filtered_regions = []
    for r in ocr_regions:
        rx1, ry1, rx2, ry2 = r["bbox_frac"]
        
        if card_bbox_frac:
            cx1, cy1, cx2, cy2 = card_bbox_frac
            inter_x = max(0, min(rx2, cx2) - max(rx1, cx1))
            inter_y = max(0, min(ry2, cy2) - max(ry1, cy1))
            inter_area = inter_x * inter_y
            region_area = (rx2 - rx1) * (ry2 - ry1)
            if region_area > 0 and (inter_area / region_area) > 0.4:
                continue # Skip card text only if card was found

        if r.get("text", "").strip():
            filtered_regions.append(r)

    # ... [bounding box logic] ...

    # 2. Compute metrics safely when px_per_cm is missing
    bar_px_length = (px_per_cm * CALIB_BAR_CM) if (px_per_cm and CALIB_BAR_CM) else None

    for idx, region in enumerate(filtered_regions, start=1):
        x1_f, y1_f, x2_f, y2_f = region["bbox_frac"]
        box_px = [int(x1_f * W), int(y1_f * H), int(x2_f * W), int(y2_f * H)]
        
        height_px = box_px[3] - box_px[1]
        width_px  = box_px[2] - box_px[0]

        # These will evaluate to None if px_per_cm is None
        height_cm = (height_px / px_per_cm) if px_per_cm else None
        visual_angle_deg = compute_visual_angle_deg(height_cm, distance_cm) if height_cm else None
        relative_ratio_to_bar = round(height_px / bar_px_length, 4) if bar_px_length else None

        result["text_blocks"].append({
            "label":                 f"text{idx}",
            "text":                  region["text"],
            "bbox_frac":             [round(c, 4) for c in [x1_f, y1_f, x2_f, y2_f]],
            "bbox_px":               box_px,
            "height_px":             height_px,
            "width_px":              width_px,
            "height_cm":             round(height_cm, 3) if height_cm else None,
            "visual_angle_deg":      visual_angle_deg,
            "relative_ratio_to_bar": relative_ratio_to_bar,
        })


    from stage2_stage3_annotator import process_and_annotate_record

    def process_stage3(
        response_id: str,
        image_path: Path,
        output_annotated_path: Path,
        survey_row: dict,
        stage2_result: dict,
        ocr_detections: list[dict]
    ) -> dict:
        """
        Executes Stage 3 processing without cropping, produces annotated image with 
        dashboard banner, and generates flattened CSV metrics dictionary.
        """
        csv_record = process_and_annotate_record(
            response_id=response_id,
            image_path=image_path,
            output_image_path=output_annotated_path,
            survey_row=survey_row,
            stage2_card_result=stage2_result,
            detected_texts_ocr=ocr_detections
        )
        
        return csv_record

    return result