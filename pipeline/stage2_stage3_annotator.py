import math
import logging
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

# ── Math Helper: Visual Angle & Relative Ratios ────────────────────────────────

def compute_visual_metrics(
    text_bbox_px: tuple[int, int, int, int], 
    bar_height_px: float, 
    px_per_cm: float,
    viewing_distance_cm: float = 40.0
) -> dict:
    """
    Computes text height in pixels/cm, physical visual angle (degrees), 
    and relative ratio of text height to calibration bar height.
    """
    _, y1, _, y2 = text_bbox_px
    h_px = abs(y2 - y1)
    
    # Relative Visual Ratio = (text height in px) / (calibration bar height/width in px)
    rel_ratio = h_px / bar_height_px if bar_height_px and bar_height_px > 0 else None
    
    # Absolute Physical Visual Angle: 2 * arctan(height_cm / (2 * distance_cm))
    abs_angle_deg = None
    if px_per_cm and px_per_cm > 0:
        h_cm = h_px / px_per_cm
        abs_angle_deg = math.degrees(2 * math.atan(h_cm / (2 * viewing_distance_cm)))
        
    return {
        "height_px": h_px,
        "relative_ratio_to_bar": round(rel_ratio, 4) if rel_ratio else None,
        "visual_angle_deg": round(abs_angle_deg, 4) if abs_angle_deg else None,
    }

def compute_contrast(img_np: np.ndarray, bbox_px: tuple[int, int, int, int]) -> float:
    """Calculates Michelson Contrast inside a bounding box."""
    x1, y1, x2, y2 = bbox_px
    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    i_max, i_min = float(np.max(gray)), float(np.min(gray))
    if i_max + i_min == 0:
        return 0.0
    return round((i_max - i_min) / (i_max + i_min), 3)


# ── Full Image Canvas Annotator & Dashboard Extension ─────────────────────────

def draw_annotations_and_dashboard(
    image: Image.Image,
    card_info: dict,
    text_regions: list[dict],
    survey_data: dict,
    output_path: Path
):
    """
    Annotates full image with bboxes for Card & Texts, extends image at bottom 
    with a dark metrics panel showing survey answers and calculated metrics.
    """
    W, H = image.size
    
    # Convert PIL Image to OpenCV (BGR) for crisp box overlays
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 1. Draw Reference Card BBox
    if card_info.get("card_found") and card_info.get("card_bbox"):
        cx1, cy1, cx2, cy2 = [
            int(card_info["card_bbox"][0] * W), int(card_info["card_bbox"][1] * H),
            int(card_info["card_bbox"][2] * W), int(card_info["card_bbox"][3] * H)
        ]
        cv2.rectangle(img_cv, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3) # Green box
        conf = card_info.get("card_confidence", 0.0)
        cv2.putText(
            img_cv, f"Ref Card ({conf})", (cx1, max(15, cy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    # 2. Draw Calibration Bar BBox
    if card_info.get("calibration_bar_bbox"):
        bx1, by1, bx2, by2 = [
            int(card_info["calibration_bar_bbox"][0] * W), int(card_info["calibration_bar_bbox"][1] * H),
            int(card_info["calibration_bar_bbox"][2] * W), int(card_info["calibration_bar_bbox"][3] * H)
        ]
        cv2.rectangle(img_cv, (bx1, by1), (bx2, by2), (255, 255, 0), 2) # Cyan box

    # 3. Draw Text Regions (Text1, Text2, ...)
    for idx, t_info in enumerate(text_regions, 1):
        x1, y1, x2, y2 = t_info["bbox_px"]
        label = f"Text{idx}"
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 215, 255), 2) # Yellow/Orange box
        cv2.putText(
            img_cv, label, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1
        )

    # Convert back to PIL to render extended canvas and clear text banner
    annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # 4. Create Extended Dashboard at the Bottom
    banner_height = 240
    new_canvas = Image.new("RGB", (W, H + banner_height), (24, 28, 36)) # Dark slate background
    new_canvas.paste(annotated_pil, (0, 0))

    draw = ImageDraw.Draw(new_canvas)
    
    # Basic font setup (uses default if system fonts aren't specified)
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_body = ImageFont.truetype("arial.ttf", 13)
    except IOError:
        font_title = font_body = ImageFont.load_default()

    # Column 1: Survey Responses
    col1_x = 20
    draw.text((col1_x, H + 15), "SURVEY RESPONSES", fill=(0, 215, 255), font=font_title)
    draw.text((col1_x, H + 40), f"Theme: {survey_data.get('Theme', 'N/A')}", fill=(255, 255, 255), font=font_body)
    
    q_yes_no = f"Q1: {survey_data.get('Q1', '-')} | Q2: {survey_data.get('Q2', '-')} | Q3: {survey_data.get('Q3', '-')}"
    draw.text((col1_x, H + 65), f"Yes/No Answers: {q_yes_no}", fill=(220, 220, 220), font=font_body)
    
    open_ended = str(survey_data.get("OpenEnded", "None"))[:55]
    draw.text((col1_x, H + 90), f"Comments: {open_ended}", fill=(180, 180, 180), font=font_body)

    # Column 2: Global Image & Card Metrics
    col2_x = int(W * 0.45)
    draw.text((col2_x, H + 15), "CARD METRICS", fill=(0, 255, 0), font=font_title)
    draw.text((col2_x, H + 40), f"Scale (px/cm): {card_info.get('px_per_cm', 'N/A')}", fill=(255, 255, 255), font=font_body)
    draw.text((col2_x, H + 65), f"Card Conf: {card_info.get('card_confidence', 'N/A')}", fill=(220, 220, 220), font=font_body)

    # Column 3: Text & Contrast Summary
    col3_x = int(W * 0.72)
    draw.text((col3_x, H + 15), "TEXT METRICS SUMMARY", fill=(255, 215, 0), font=font_title)
    
    y_pos = H + 40
    for idx, t_info in enumerate(text_regions[:4], 1): # Display first 4 texts in banner
        rel = t_info.get("relative_ratio_to_bar", "N/A")
        c_val = t_info.get("contrast", "N/A")
        draw.text(
            (col3_x, y_pos), 
            f"Text{idx}: RelRatio={rel} | Contrast={c_val}", 
            fill=(220, 220, 220), 
            font=font_body
        )
        y_pos += 22

    # Save final annotated canvas
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_canvas.save(output_path)
    log.info(f"Saved annotated image with extended metrics banner to {output_path}")


# ── Pipeline Orchestrator & CSV Record Generator ─────────────────────────────

def process_and_annotate_record(
    response_id: str,
    image_path: Path,
    output_image_path: Path,
    survey_row: dict,
    stage2_card_result: dict,
    detected_texts_ocr: list[dict]
) -> dict:
    """
    Master pipeline wrapper. Processes raw data without cropping, computes text visual 
    angles relative to the reference card, draws overlay banner, and formats a full CSV row.
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_np = np.array(img)

    # Calibration bar height in pixels (used for relative ratio calculation)
    bar_px = stage2_card_result.get("bar_width_px") or 1.0

    processed_texts = []
    text_csv_payload = {}

    for idx, t in enumerate(detected_texts_ocr, 1):
        # Convert fractional or raw coordinates to absolute pixels [x1, y1, x2, y2]
        x1 = int(t["bbox_frac"][0] * W) if "bbox_frac" in t else t["bbox_px"][0]
        y1 = int(t["bbox_frac"][1] * H) if "bbox_frac" in t else t["bbox_px"][1]
        x2 = int(t["bbox_frac"][2] * W) if "bbox_frac" in t else t["bbox_px"][2]
        y2 = int(t["bbox_frac"][3] * H) if "bbox_frac" in t else t["bbox_px"][3]
        
        bbox_px = (x1, y1, x2, y2)
        
        # Calculate visual angle metrics and contrast
        metrics = compute_visual_metrics(
            bbox_px, 
            bar_height_px=bar_px, 
            px_per_cm=stage2_card_result.get("px_per_cm")
        )
        contrast = compute_contrast(img_np, bbox_px)

        t_entry = {
            "text_id": f"Text{idx}",
            "text": t.get("text", ""),
            "bbox_px": bbox_px,
            "contrast": contrast,
            **metrics
        }
        processed_texts.append(t_entry)

        # Structure individual text records for direct CSV flattening
        text_csv_payload.update({
            f"text_{idx}_str": t.get("text", ""),
            f"text_{idx}_bbox_px": str(list(bbox_px)),
            f"text_{idx}_height_px": metrics["height_px"],
            f"text_{idx}_rel_ratio_to_bar": metrics["relative_ratio_to_bar"],
            f"text_{idx}_visual_angle_deg": metrics["visual_angle_deg"],
            f"text_{idx}_contrast": contrast,
        })

    # Save full visual annotated banner without cropping
    draw_annotations_and_dashboard(
        image=img,
        card_info=stage2_card_result,
        text_regions=processed_texts,
        survey_data=survey_row,
        output_path=output_image_path
    )

    # Return merged dictionary containing Survey Data + Stage 2 Card Data + Text Metrics
    combined_csv_record = {
        **survey_row,  # Includes all original survey columns
        "response_id": response_id,
        "image_width_px": W,
        "image_height_px": H,
        "card_found": stage2_card_result.get("card_found"),
        "card_confidence": stage2_card_result.get("card_confidence"),
        "card_bbox": str(stage2_card_result.get("card_bbox")),
        "calibration_bar_bbox": str(stage2_card_result.get("calibration_bar_bbox")),
        "px_per_cm": stage2_card_result.get("px_per_cm"),
        **text_csv_payload  # Adds text_1_*, text_2_*, ... columns
    }

    return combined_csv_record