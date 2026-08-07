"""
Wicked Words — Image Annotation
=================================
For each processed submission, draws bounding boxes on the image and
appends an extended bottom banner with:
  - Participant ID (P-xxx)
  - All survey answers (theme, legibility ratings, open-text responses)
  - Computed image DVs (visual angle, RMS contrast, Michelson contrast)

Run:
    python annotate.py

Output: outputs/annotated/{ResponseId}_annotated.jpg
"""

import ast
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = Path("outputs/results.csv")
IMAGE_DIR    = Path("Data/Photo Upload")
OUT_DIR      = Path("outputs/annotated")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors (RGB) ──────────────────────────────────────────────────────────────
GREEN   = (34, 197, 94)
CYAN    = (6, 182, 212)
RED     = (239, 68, 68)
WHITE   = (255, 255, 255)
BLACK   = (0, 0, 0)
DARK_BG = (30, 30, 40)
GRAY    = (180, 180, 180)
YELLOW  = (250, 204, 21)

FONT_SIZE_TITLE  = 22
FONT_SIZE_BODY   = 17
FONT_SIZE_SMALL  = 14
LINE_H           = 22   # pixels per text line in banner
BANNER_PADDING   = 18


def _font(size: int):
    """Load a font, fall back to default if not available."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _parse_bbox(val, img_w: int, img_h: int):
    """
    Parse a bbox value that may be:
      - a list [x1,y1,x2,y2] in fractional (0-1) coords
      - a string representation of the above
    Returns pixel coords (x1,y1,x2,y2) or None.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except Exception:
            return None
    if not isinstance(val, (list, tuple)) or len(val) < 4:
        return None
    x1, y1, x2, y2 = val[:4]
    # If values are fractions (0-1), convert to pixels
    if max(x1, y1, x2, y2) <= 1.0:
        x1, y1, x2, y2 = x1*img_w, y1*img_h, x2*img_w, y2*img_h
    return int(x1), int(y1), int(x2), int(y2)


def _draw_box(draw: ImageDraw.Draw, bbox, color, label: str, font):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    # Label background
    tw = draw.textlength(label, font=font)
    draw.rectangle([x1, y1 - 22, x1 + tw + 6, y1], fill=color)
    draw.text((x1 + 3, y1 - 20), label, fill=BLACK, font=font)


def _fmt(val, decimals=3):
    """Format a value for display — handles None, NaN, float, bool."""
    if val is None:
        return "—"
    if isinstance(val, float) and np.isnan(val):
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _wrap(text: str, width: int = 80) -> list[str]:
    """Wrap long text into lines."""
    if not text or text == "—":
        return [text]
    return textwrap.wrap(str(text), width=width) or ["—"]


def build_banner(row: pd.Series, img_w: int, participant_id: str) -> Image.Image:
    """
    Build the extended bottom banner as a PIL Image.
    Two-column layout: left = survey answers, right = image metrics.
    """
    col_w = img_w // 2

    # ── Collect survey answers ────────────────────────────────────────────────
    theme     = _fmt(row.get("survey_theme", row.get("Theme Choice")))
    legible   = _fmt(row.get("legible",      row.get("Experience 1")))
    easy_read = _fmt(row.get("easy_to_read", row.get("Experience 2")))
    prolonged = _fmt(row.get("prolonged_ok", row.get("Experience 3")))
    vision    = _fmt(row.get("vision_correction", row.get("Experience 4")))
    challenge = _fmt(row.get("challenge_text",    row.get("Follow up 1")))
    env_text  = _fmt(row.get("environment_text",  row.get("Follow up 2")))
    personal  = _fmt(row.get("personal_factor_text", row.get("Follow up 3")))
    feedback  = _fmt(row.get("feedback",         row.get("Feedback Question")))

    # ── Collect image metrics ─────────────────────────────────────────────────
    va_deg    = _fmt(row.get("visual_angle_deg"), 4)
    va_calib  = _fmt(row.get("visual_angle_calib_multiples"), 4)
    logmar    = _fmt(row.get("logmar"), 3)
    snellen   = _fmt(row.get("snellen_equiv"))
    rms       = _fmt(row.get("rms_contrast"), 4)
    michelson = _fmt(row.get("michelson_contrast"), 4)
    mad       = _fmt(row.get("mad_contrast"), 4)
    px_per_cm = _fmt(row.get("px_per_cm"), 2)
    cap_h_cm  = _fmt(row.get("cap_height_cm"), 3)
    human_pred = _fmt(row.get("human_behavior_pred"))
    difficulty = _fmt(row.get("human_difficulty_score"), 1)

    # ── Build left column lines ───────────────────────────────────────────────
    left_lines: list[tuple] = []   # (text, color, font_size, indent)

    left_lines.append((f"PARTICIPANT: {participant_id}  |  ID: {_fmt(row.get('response_id'))}", YELLOW, FONT_SIZE_TITLE, 0))
    left_lines.append(("── SURVEY ANSWERS ──────────────────────────────", GRAY, FONT_SIZE_SMALL, 0))
    left_lines.append((f"Theme:          {theme}", WHITE, FONT_SIZE_BODY, 0))
    left_lines.append((f"Legible:        {legible}", WHITE, FONT_SIZE_BODY, 0))
    left_lines.append((f"Easy to read:   {easy_read}", WHITE, FONT_SIZE_BODY, 0))
    left_lines.append((f"Prolonged OK:   {prolonged}", WHITE, FONT_SIZE_BODY, 0))
    left_lines.append((f"Vision corr:    {vision}", WHITE, FONT_SIZE_BODY, 0))
    left_lines.append(("── OPEN TEXT ───────────────────────────────────", GRAY, FONT_SIZE_SMALL, 0))
    for line in _wrap(f"Challenge: {challenge}", 55):
        left_lines.append((line, GRAY, FONT_SIZE_SMALL, 0))
    for line in _wrap(f"Environment: {env_text}", 55):
        left_lines.append((line, GRAY, FONT_SIZE_SMALL, 0))
    for line in _wrap(f"Personal: {personal}", 55):
        left_lines.append((line, GRAY, FONT_SIZE_SMALL, 0))
    for line in _wrap(f"Feedback: {feedback}", 55):
        left_lines.append((line, GRAY, FONT_SIZE_SMALL, 0))

    # ── Build right column lines ──────────────────────────────────────────────
    right_lines: list[tuple] = []

    right_lines.append(("── IMAGE METRICS ───────────────────────────────", GRAY, FONT_SIZE_SMALL, 0))
    right_lines.append((f"Visual angle:        {va_deg} °", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"VA / calib bar:      {va_calib}×", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"logMAR:              {logmar}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"Snellen equiv:       {snellen}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"Cap height (cm):     {cap_h_cm}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"px / cm:             {px_per_cm}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append(("── CONTRAST ────────────────────────────────────", GRAY, FONT_SIZE_SMALL, 0))
    right_lines.append((f"Michelson contrast:  {michelson}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"RMS contrast:        {rms}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"MAD contrast:        {mad}", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append(("── PREDICTION ──────────────────────────────────", GRAY, FONT_SIZE_SMALL, 0))
    right_lines.append((f"Human difficulty:    {difficulty} / 100", WHITE, FONT_SIZE_BODY, 0))
    right_lines.append((f"Prediction:          {human_pred}", YELLOW, FONT_SIZE_BODY, 0))

    # ── Compute banner height ─────────────────────────────────────────────────
    n_lines = max(len(left_lines), len(right_lines))
    banner_h = BANNER_PADDING * 2 + n_lines * LINE_H + 10

    banner = Image.new("RGB", (img_w, banner_h), DARK_BG)
    draw = ImageDraw.Draw(banner)

    # Divider line at top of banner
    draw.line([(0, 2), (img_w, 2)], fill=YELLOW, width=2)
    # Vertical divider between columns
    draw.line([(col_w, 8), (col_w, banner_h - 8)], fill=(70, 70, 80), width=1)

    def render_lines(lines, x_start):
        y = BANNER_PADDING
        for text, color, fsize, indent in lines:
            font = _font(fsize)
            draw.text((x_start + indent, y), text, fill=color, font=font)
            y += LINE_H

    render_lines(left_lines, BANNER_PADDING)
    render_lines(right_lines, col_w + BANNER_PADDING)

    return banner


def annotate_image(row: pd.Series, participant_id: str) -> bool:
    """Annotate one image. Returns True if saved successfully."""
    rid = str(row.get("response_id", ""))
    if not rid:
        return False

    # Find source image
    matches = list(IMAGE_DIR.glob(f"{rid}_*"))
    if not matches:
        print(f"  SKIP {rid}: image not found")
        return False

    img = Image.open(matches[0]).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font_label = _font(FONT_SIZE_SMALL)

    # Draw card bbox (green)
    card_bbox = _parse_bbox(row.get("card_bbox"), W, H)
    _draw_box(draw, card_bbox, GREEN, "Ref Card", font_label)

    # Draw calibration bar bbox (cyan)
    bar_bbox = _parse_bbox(row.get("calibration_bar_bbox"), W, H)
    if bar_bbox is not None:
        bx1, by1, bx2, by2 = bar_bbox
        # Ensure bar is at least 4px tall so it's always visible
        if by2 - by1 < 4:
            mid = (by1 + by2) // 2
            by1, by2 = mid - 2, mid + 2
        draw.rectangle([bx1, by1, bx2, by2], outline=CYAN, width=3)
        # Label below the bar (avoids going off-screen when bar is near top)
        font_label = _font(FONT_SIZE_SMALL)
        bar_w_px = row.get("bar_width_px")
        label = f"Calib Bar 8cm ({int(bar_w_px)}px)" if bar_w_px and not (isinstance(bar_w_px, float) and np.isnan(bar_w_px)) else "Calib Bar 8cm"
        draw.text((bx1 + 4, by2 + 4), label, fill=CYAN, font=font_label)

    # Draw target text bbox (yellow)
    target_bbox = _parse_bbox(row.get("target_bbox"), W, H)
    _draw_box(draw, target_bbox, YELLOW, "Target Text", font_label)

    # Build and attach banner
    banner = build_banner(row, W, participant_id)
    combined = Image.new("RGB", (W, H + banner.height))
    combined.paste(img, (0, 0))
    combined.paste(banner, (0, H))

    out_path = OUT_DIR / f"{rid}_annotated.jpg"
    combined.save(out_path, quality=92)
    return True


def main():
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run the pipeline first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows from results.csv")

    # Build P-xxx token map
    if "participant_email_anon" in df.columns:
        unique_emails = sorted(df["participant_email_anon"].dropna().unique())
        token_map = {e: f"P-{i+1:03d}" for i, e in enumerate(unique_emails)}
        df["participant_id"] = df["participant_email_anon"].map(token_map).fillna("P-???")
    else:
        df["participant_id"] = [f"P-{i+1:03d}" for i in range(len(df))]

    saved, skipped = 0, 0
    for _, row in df.iterrows():
        pid = row.get("participant_id", "P-???")
        ok = annotate_image(row, pid)
        if ok:
            saved += 1
            print(f"  ✓ {row.get('response_id')}  ({pid})")
        else:
            skipped += 1

    print(f"\n✓ Annotated {saved} images → {OUT_DIR}")
    if skipped:
        print(f"  {skipped} skipped (no image on disk)")


if __name__ == "__main__":
    main()
