import math
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

# ── Theme → screen type detection ─────────────────────────────────────────────
_DIGITAL_SCREEN_KEYWORDS = [
    "mini screen", "miniscreen", "monday", "digital", "screen", "display",
    "monitor", "phone", "tablet", "computer",
]

def _is_digital_screen(survey_data: dict) -> bool:
    theme = str(
        survey_data.get("survey_theme", survey_data.get("Theme Choice", ""))
    ).lower()
    text_theme = str(survey_data.get("text_theme", "")).lower()
    return (
        any(k in theme for k in _DIGITAL_SCREEN_KEYWORDS)
        or text_theme == "digital_screen"
    )


# ── Contrast helpers ───────────────────────────────────────────────────────────

def _linearise(c: float) -> float:
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _relative_luminance(rgb: tuple) -> float:
    r, g, b = [_linearise(float(v)) for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def compute_michelson(crop_gray: np.ndarray) -> float:
    i_max, i_min = float(np.max(crop_gray)), float(np.min(crop_gray))
    if i_max + i_min == 0:
        return 0.0
    return round((i_max - i_min) / (i_max + i_min), 3)

def compute_rms(crop_gray: np.ndarray) -> float:
    return round(float(crop_gray.std()) / 128.0, 4)

def compute_wcag(crop_rgb: np.ndarray) -> float:
    """
    WCAG 2.1 contrast ratio between median bright and median dark pixel clusters.
    Only meaningful for digital screen content.
    """
    flat = crop_rgb.reshape(-1, 3).astype(float)
    lums = np.array([_relative_luminance(px) for px in flat])
    bright = float(np.percentile(lums, 90))
    dark   = float(np.percentile(lums, 10))
    L1, L2 = max(bright, dark), min(bright, dark)
    return round((L1 + 0.05) / (L2 + 0.05), 2)

def compute_contrast_metrics(
    img_np: np.ndarray,
    bbox_px: tuple,
    is_digital: bool,
) -> dict:
    x1, y1, x2, y2 = bbox_px
    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return {"michelson": None, "rms": None, "wcag": None}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    result = {
        "michelson": compute_michelson(gray),
        "rms":       compute_rms(gray),
        "wcag":      compute_wcag(crop) if is_digital else None,
    }
    return result


# ── Text height vs box height ──────────────────────────────────────────────────

def estimate_text_height(
    img_np: np.ndarray,
    bbox_px: tuple,
) -> dict:
    """
    Separate text (ink) height from the full bounding box height.

    For a horizontal text box the ink rows are those whose mean luminance
    is significantly darker than the background.  For a vertical/rotated
    box the difference between box_height and text_height can be large.

    Returns:
        box_height_px   — full bbox height (y2 - y1)
        text_height_px  — estimated cap-height of the ink strokes
        height_diff_px  — box_height - text_height  (padding / orientation gap)
        orientation     — "horizontal" | "vertical" (aspect-ratio heuristic)
    """
    x1, y1, x2, y2 = bbox_px
    box_h = abs(y2 - y1)
    box_w = abs(x2 - x1)

    orientation = "vertical" if box_h > box_w * 1.5 else "horizontal"

    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            "box_height_px": box_h, "text_height_px": None,
            "height_diff_px": None, "orientation": orientation,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(float)

    if orientation == "horizontal":
        # Row-wise: find rows that are darker than background median
        bg_lum   = float(np.percentile(gray, 80))   # bright rows = background
        row_mean = gray.mean(axis=1)
        ink_rows = np.where(row_mean < bg_lum * 0.75)[0]
    else:
        # Column-wise for vertical text
        bg_lum   = float(np.percentile(gray, 80))
        col_mean = gray.mean(axis=0)
        ink_rows = np.where(col_mean < bg_lum * 0.75)[0]

    if len(ink_rows) >= 2:
        text_h = int(ink_rows[-1] - ink_rows[0] + 1)
    else:
        text_h = box_h   # fallback: can't separate

    return {
        "box_height_px":  box_h,
        "text_height_px": text_h,
        "height_diff_px": box_h - text_h,
        "orientation":    orientation,
    }


# ── Visual angle & relative ratio ─────────────────────────────────────────────

def compute_visual_metrics(
    text_bbox_px: tuple,
    bar_width_px: float,        # ← width of the 8 cm calibration bar
    px_per_cm: float,
    viewing_distance_cm: float = 40.0,
) -> dict:
    """
    Uses TEXT height (not box height) for visual angle.
    Relative ratio = text_height_px / bar_width_px  (bar width = 8 cm reference).
    """
    x1, y1, x2, y2 = text_bbox_px
    box_h = abs(y2 - y1)

    # Visual angle uses box height as upper bound; text height computed separately
    abs_angle_deg = None
    if px_per_cm and px_per_cm > 0:
        h_cm = box_h / px_per_cm
        abs_angle_deg = round(
            math.degrees(2 * math.atan(h_cm / (2 * viewing_distance_cm))), 4
        )

    # Relative ratio: text box height as fraction of the 8 cm bar width
    rel_ratio = round(box_h / bar_width_px, 4) if bar_width_px and bar_width_px > 0 else None

    return {
        "box_height_px":   box_h,
        "visual_angle_deg": abs_angle_deg,
        "relative_ratio_to_bar_width": rel_ratio,
    }


# ── Font loader ────────────────────────────────────────────────────────────────

def _font(size: int):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Main drawing function ──────────────────────────────────────────────────────

def draw_annotations_and_dashboard(
    image: Image.Image,
    card_info: dict,
    text_regions: list[dict],
    survey_data: dict,
    output_path: Path,
):
    W, H = image.size
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    is_digital = _is_digital_screen(survey_data)

    # ── 1. Reference card — GREEN ─────────────────────────────────────────────
    if card_info.get("card_found") and card_info.get("card_bbox"):
        cx1, cy1, cx2, cy2 = [
            int(card_info["card_bbox"][0] * W), int(card_info["card_bbox"][1] * H),
            int(card_info["card_bbox"][2] * W), int(card_info["card_bbox"][3] * H),
        ]
        cv2.rectangle(img_cv, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
        cv2.putText(img_cv, "Ref Card", (cx1, max(18, cy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # ── 2. Calibration bar — MAGENTA, clearly labelled with measured width ────
    if card_info.get("calibration_bar_bbox"):
        bx1, by1, bx2, by2 = [
            int(card_info["calibration_bar_bbox"][0] * W),
            int(card_info["calibration_bar_bbox"][1] * H),
            int(card_info["calibration_bar_bbox"][2] * W),
            int(card_info["calibration_bar_bbox"][3] * H),
        ]
        bar_w_px = card_info.get("bar_width_px") or (bx2 - bx1)
        cv2.rectangle(img_cv, (bx1, by1), (bx2, by2), (255, 0, 255), 3)  # Magenta
        # Horizontal arrow showing bar width
        mid_y = (by1 + by2) // 2
        cv2.arrowedLine(img_cv, (bx1, mid_y), (bx2, mid_y), (255, 0, 255), 2, tipLength=0.03)
        cv2.arrowedLine(img_cv, (bx2, mid_y), (bx1, mid_y), (255, 0, 255), 2, tipLength=0.03)
        label = f"Calib Bar  {bar_w_px:.0f}px = 8cm"
        cv2.putText(img_cv, label, (bx1, max(18, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # ── 3. Text regions — CYAN box + separate text-height line ───────────────
    for idx, t in enumerate(text_regions, 1):
        x1, y1, x2, y2 = t["bbox_px"]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 215, 255), 2)

        # Draw a separate horizontal line at estimated text (ink) height
        text_h = t.get("text_height_px")
        if text_h and text_h < (y2 - y1):
            ink_top = y2 - text_h   # ink starts from bottom of box upward
            cv2.line(img_cv, (x1, ink_top), (x2, ink_top), (0, 140, 255), 1)
            cv2.putText(img_cv, f"ink~{text_h}px", (x2 + 4, ink_top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        cv2.putText(img_cv, f"Text{idx}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

    annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # ── 4. Banner ─────────────────────────────────────────────────────────────
    banner_h = 290
    canvas = Image.new("RGB", (W, H + banner_h), (24, 28, 36))
    canvas.paste(annotated_pil, (0, 0))
    draw = ImageDraw.Draw(canvas)

    ft = _font(16)
    fb = _font(13)

    # Divider
    draw.line([(0, H + 2), (W, H + 2)], fill=(80, 80, 100), width=2)

    # ── Col 1: Survey answers ─────────────────────────────────────────────────
    x1c = 20
    draw.text((x1c, H + 12), "SURVEY RESPONSES", fill=(0, 215, 255), font=ft)

    theme     = str(survey_data.get("survey_theme", survey_data.get("Theme Choice", "N/A")))
    legible   = survey_data.get("legible",      survey_data.get("Experience 1", "-"))
    easy_read = survey_data.get("easy_to_read", survey_data.get("Experience 2", "-"))
    prolonged = survey_data.get("prolonged_ok", survey_data.get("Experience 3", "-"))
    challenge = str(survey_data.get("challenge_text",       survey_data.get("Follow up 1", "")) or "")[:72]
    env_text  = str(survey_data.get("environment_text",     survey_data.get("Follow up 2", "")) or "")[:72]
    personal  = str(survey_data.get("personal_factor_text", survey_data.get("Follow up 3", "")) or "")[:72]

    draw.text((x1c, H + 36),  f"Theme:       {theme}",                                    fill=(255, 255, 255), font=fb)
    draw.text((x1c, H + 56),  f"Legible: {legible}  |  Easy: {easy_read}  |  Prolonged: {prolonged}", fill=(220, 220, 220), font=fb)
    draw.text((x1c, H + 76),  f"Challenge:   {challenge}",  fill=(180, 180, 180), font=fb)
    draw.text((x1c, H + 96),  f"Environment: {env_text}",   fill=(180, 180, 180), font=fb)
    draw.text((x1c, H + 116), f"Personal:    {personal}",   fill=(180, 180, 180), font=fb)

    # ── Col 2: Card metrics ───────────────────────────────────────────────────
    x2c = int(W * 0.44)
    draw.text((x2c, H + 12), "CARD METRICS", fill=(0, 255, 0), font=ft)
    draw.text((x2c, H + 36), f"px / cm:      {card_info.get('px_per_cm', 'N/A')}",  fill=(255, 255, 255), font=fb)
    draw.text((x2c, H + 56), f"Bar width:    {card_info.get('bar_width_px', 'N/A'):.0f} px = 8 cm"
              if isinstance(card_info.get("bar_width_px"), float) else
              f"Bar width:    {card_info.get('bar_width_px', 'N/A')} px = 8 cm",
              fill=(255, 0, 255), font=fb)
    draw.text((x2c, H + 76), f"Confidence:   {card_info.get('card_confidence', 'N/A')}", fill=(220, 220, 220), font=fb)
    draw.text((x2c, H + 96),fill=(250, 204, 21), font=fb)

    # ── Col 3: Text metrics ───────────────────────────────────────────────────
    x3c = int(W * 0.72)
    draw.text((x3c, H + 12), "TEXT METRICS", fill=(255, 215, 0), font=ft)

    y_pos = H + 36
    for idx, t in enumerate(text_regions[:5], 1):
        box_h  = t.get("box_height_px",  "N/A")
        text_h = t.get("text_height_px", "N/A")
        diff   = t.get("height_diff_px", "N/A")
        orient = t.get("orientation", "")
        rel    = t.get("relative_ratio_to_bar_width", "N/A")
        va     = t.get("visual_angle_deg", "N/A")
        mich   = t.get("michelson", "N/A")
        rms    = t.get("rms", "N/A")
        wcag   = t.get("wcag")

        draw.text((x3c, y_pos),
                  f"Text{idx} [{orient}]  box={box_h}px  ink={text_h}px  diff={diff}px",
                  fill=(220, 220, 220), font=fb)
        y_pos += 18
        draw.text((x3c, y_pos),
                  f"  VA={va}°  ratio={rel}  Mich={mich}  RMS={rms}",
                  fill=(180, 180, 180), font=fb)
        y_pos += 18
        if wcag is not None:
            wcag_color = (34, 197, 94) if wcag >= 4.5 else (239, 68, 68)
            draw.text((x3c, y_pos),
                      f"  WCAG={wcag}:1  {'✓ AA' if wcag >= 4.5 else '✗ fails AA'}",
                      fill=wcag_color, font=fb)
            y_pos += 18
        y_pos += 4   # gap between text entries

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    log.info(f"Saved annotated image → {output_path}")


# ── Pipeline orchestrator ──────────────────────────────────────────────────────

def process_and_annotate_record(
    response_id: str,
    image_path: Path,
    output_image_path: Path,
    survey_row: dict,
    stage2_card_result: dict,
    detected_texts_ocr: list[dict],
) -> dict:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    img_np = np.array(img)

    is_digital = _is_digital_screen(survey_row)
    bar_width_px = stage2_card_result.get("bar_width_px") or 1.0

    processed_texts = []
    text_csv_payload = {}

    for idx, t in enumerate(detected_texts_ocr, 1):
        if "bbox_frac" in t:
            x1 = int(t["bbox_frac"][0] * W)
            y1 = int(t["bbox_frac"][1] * H)
            x2 = int(t["bbox_frac"][2] * W)
            y2 = int(t["bbox_frac"][3] * H)
        else:
            x1, y1, x2, y2 = t["bbox_px"]
        bbox_px = (x1, y1, x2, y2)

        vm      = compute_visual_metrics(bbox_px, bar_width_px, stage2_card_result.get("px_per_cm"))
        heights = estimate_text_height(img_np, bbox_px)
        contrast = compute_contrast_metrics(img_np, bbox_px, is_digital)

        entry = {
            "text_id": f"Text{idx}",
            "text":    t.get("text", ""),
            "bbox_px": bbox_px,
            **vm,
            **heights,
            **contrast,
        }
        processed_texts.append(entry)

        text_csv_payload.update({
            f"text_{idx}_str":                    t.get("text", ""),
            f"text_{idx}_bbox_px":                str(list(bbox_px)),
            f"text_{idx}_box_height_px":          heights["box_height_px"],
            f"text_{idx}_text_height_px":         heights["text_height_px"],
            f"text_{idx}_height_diff_px":         heights["height_diff_px"],
            f"text_{idx}_orientation":            heights["orientation"],
            f"text_{idx}_visual_angle_deg":       vm["visual_angle_deg"],
            f"text_{idx}_rel_ratio_to_bar_width": vm["relative_ratio_to_bar_width"],
            f"text_{idx}_michelson":              contrast["michelson"],
            f"text_{idx}_rms":                    contrast["rms"],
            f"text_{idx}_wcag":                   contrast["wcag"],
        })

    draw_annotations_and_dashboard(
        image=img,
        card_info=stage2_card_result,
        text_regions=processed_texts,
        survey_data=survey_row,
        output_path=output_image_path,
    )

    return {
        **survey_row,
        "response_id":          response_id,
        "image_width_px":       W,
        "image_height_px":      H,
        "card_found":           stage2_card_result.get("card_found"),
        "card_confidence":      stage2_card_result.get("card_confidence"),
        "card_bbox":            str(stage2_card_result.get("card_bbox")),
        "calibration_bar_bbox": str(stage2_card_result.get("calibration_bar_bbox")),
        "bar_width_px":         bar_width_px,
        "px_per_cm":            stage2_card_result.get("px_per_cm"),
        **text_csv_payload,
    }
