"""
Stage 4 — Metric Computation
==============================
Converts pixel-level measurements into perceptually meaningful units
using the calibration scale factor from Stage 2.

Metrics computed:
  Visual size
  ─────────────
  • visual_angle_deg   : angular size of the text at standard viewing distance
  • logmar             : log10 of visual angle in arcminutes (clinical standard)
  • snellen_equiv      : approximate Snellen acuity (e.g. "20/80")

  Contrast
  ─────────
  • michelson_contrast : (Lmax - Lmin) / (Lmax + Lmin) — global range
  • rms_contrast       : std(luminance) / 128 — overall variability
  • mad_contrast       : median absolute deviation — robust to glare/outliers
  • weber_contrast     : (Ltext - Lbg) / Lbg — local text-on-background

  From Gemini RGB estimates (Stage 3)
  ────────────────────────────────────
  • relative_luminance_text : WCAG relative luminance of text color
  • relative_luminance_bg   : WCAG relative luminance of background
  • wcag_contrast_ratio     : (L_lighter + 0.05) / (L_darker + 0.05)
    WCAG AA requires >= 4.5:1 for normal text, >= 3:1 for large text.
"""

import math
import logging
import numpy as np
from pathlib import Path
from PIL import Image

from config import VIEWING_DISTANCE_CM, CALIB_BAR_CM

log = logging.getLogger(__name__)


# ── Visual angle ──────────────────────────────────────────────────────────────

def visual_angle_deg(cap_height_px: float, px_per_cm: float,
                     viewing_dist_cm: float = VIEWING_DISTANCE_CM) -> float:
    """
    Compute visual angle subtended by the text cap-height.
    Formula: 2 * arctan(height_cm / (2 * viewing_distance_cm))
    """
    height_cm = cap_height_px / px_per_cm
    angle_rad = 2 * math.atan(height_cm / (2 * viewing_dist_cm))
    return math.degrees(angle_rad)


def logmar_from_deg(va_deg: float) -> float:
    """
    LogMAR = log10(visual angle in arcminutes).
    1 arcminute = 0.0167°; 20/20 acuity corresponds to 1 arcmin (logMAR 0.0).
    """
    va_arcmin = va_deg * 60.0
    if va_arcmin <= 0:
        return float("nan")
    return round(math.log10(va_arcmin), 3)


def snellen_from_logmar(logmar: float) -> str:
    """
    Approximate Snellen denominator from logMAR value.
    Snellen = 20 / (20 * 10^logMAR) → denominator = 20 * 10^logMAR.
    """
    if math.isnan(logmar):
        return "N/A"
    denom = 20 * (10 ** logmar)
    # Round to nearest standard Snellen value
    standards = [10, 12, 15, 20, 25, 30, 40, 50, 63, 80, 100, 125, 160, 200, 250, 320, 400]
    closest = min(standards, key=lambda s: abs(s - denom))
    return f"20/{closest}"


def compute_size_metrics(cap_height_px: float, px_per_cm: float) -> dict:
    if cap_height_px is None or px_per_cm is None or px_per_cm <= 0:
        return {"visual_angle_deg": None, "logmar": None, "snellen_equiv": None,
                "cap_height_cm": None}

    va = visual_angle_deg(cap_height_px, px_per_cm)
    lm = logmar_from_deg(va)
    return {
        "visual_angle_deg": round(va, 4),
        "logmar": lm,
        "snellen_equiv": snellen_from_logmar(lm),
        "cap_height_cm": round(cap_height_px / px_per_cm, 3),
    }


# ── Pixel-level contrast (from target crop) ────────────────────────────────────

def compute_pixel_contrast(target_crop_path: Path) -> dict:
    """
    Compute contrast metrics from the saved target crop image.
    All metrics are computed on the grayscale luminance channel.
    """
    empty = {
        "michelson_contrast": None,
        "rms_contrast": None,
        "mad_contrast": None,
        "mean_luminance": None,
    }
    if not target_crop_path.exists():
        return empty

    img = Image.open(target_crop_path).convert("L")
    arr = np.array(img, dtype=float)

    if arr.size == 0:
        return empty

    L_max = arr.max()
    L_min = arr.min()

    # Michelson contrast: overall luminance range
    michelson = (L_max - L_min) / (L_max + L_min + 1e-6)

    # RMS contrast: standard deviation normalized to [0, 1]
    rms = arr.std() / 128.0

    # MAD contrast: robust to specular highlights / glare
    mad = float(np.median(np.abs(arr - np.median(arr)))) / 128.0

    return {
        "michelson_contrast": round(michelson, 4),
        "rms_contrast": round(rms, 4),
        "mad_contrast": round(mad, 4),
        "mean_luminance": round(float(arr.mean()), 2),
    }


# ── WCAG contrast (from Gemini RGB estimates) ─────────────────────────────────

def _srgb_to_linear(c: float) -> float:
    """Convert sRGB channel [0,255] to linear light."""
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: list[int]) -> float:
    """WCAG 2.1 relative luminance from sRGB triple [R, G, B] in 0–255."""
    r, g, b = [_srgb_to_linear(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def wcag_contrast_ratio(text_rgb: list[int], bg_rgb: list[int]) -> float:
    """WCAG 2.1 contrast ratio. Range: 1:1 (no contrast) to 21:1 (black on white)."""
    l1 = relative_luminance(text_rgb)
    l2 = relative_luminance(bg_rgb)
    lighter = max(l1, l2)
    darker  = min(l1, l2)
    return round((lighter + 0.05) / (darker + 0.05), 3)


def wcag_compliance(ratio: float) -> dict:
    """Assess WCAG 2.1 AA and AAA compliance."""
    return {
        "wcag_aa_normal":  ratio >= 4.5,   # normal text
        "wcag_aa_large":   ratio >= 3.0,   # large text (≥18pt or ≥14pt bold)
        "wcag_aaa_normal": ratio >= 7.0,
        "wcag_aaa_large":  ratio >= 4.5,
    }


def compute_color_contrast(text_rgb: list | None, bg_rgb: list | None) -> dict:
    empty = {
        "relative_luminance_text": None,
        "relative_luminance_bg": None,
        "wcag_contrast_ratio": None,
        "wcag_aa_normal": None,
        "wcag_aa_large": None,
    }
    if text_rgb is None or bg_rgb is None:
        return empty

    ratio = wcag_contrast_ratio(text_rgb, bg_rgb)
    compliance = wcag_compliance(ratio)
    return {
        "relative_luminance_text": round(relative_luminance(text_rgb), 4),
        "relative_luminance_bg": round(relative_luminance(bg_rgb), 4),
        "wcag_contrast_ratio": ratio,
        **compliance,
    }


# ── Readability prediction ───────────────────────────────────────────────────

# Human acuity reference points (logMAR → difficulty weight)
# Based on clinical vision standards:
#   logMAR 0.0  = 20/20  perfect acuity  → weight 1.0 (no extra difficulty)
#   logMAR 0.3  = 20/40  mild impairment → weight 1.3
#   logMAR 1.0  = 20/200 legal blindness → weight 2.5
ACUITY_BREAKPOINTS = [
    (0.0,  1.0),
    (0.3,  1.3),
    (0.5,  1.6),
    (1.0,  2.5),
    (2.0,  4.0),
]


def _acuity_weight(logmar: float) -> float:
    """Interpolate difficulty weight from logMAR value."""
    if logmar is None or math.isnan(logmar):
        return 1.0
    for i in range(len(ACUITY_BREAKPOINTS) - 1):
        lm0, w0 = ACUITY_BREAKPOINTS[i]
        lm1, w1 = ACUITY_BREAKPOINTS[i + 1]
        if lm0 <= logmar <= lm1:
            t = (logmar - lm0) / (lm1 - lm0)
            return round(w0 + t * (w1 - w0), 3)
    return ACUITY_BREAKPOINTS[-1][1]


def predict_text_behavior(visual_angle: float | None, logmar: float | None,
                          wcag_ratio: float | None,
                          michelson: float | None) -> dict:
    """
    Predict whether the text environment is objectively readable.
    Based purely on physical/optical metrics — no human input.

    Rules derived from vision science literature:
      - Visual angle < 0.1° → too small for most people (below acuity limit)
      - Visual angle 0.1–0.5° → borderline / requires effort
      - Visual angle > 0.5° → comfortably large
      - WCAG ratio < 3.0 → poor contrast (fails all accessibility standards)
      - WCAG ratio 3.0–4.5 → marginal (passes large text only)
      - WCAG ratio >= 4.5 → adequate contrast
      - Michelson < 0.3 → low pixel contrast
    """
    issues = []
    score = 100  # start at 100, deduct for each issue

    # Size assessment
    if visual_angle is not None:
        if visual_angle < 0.1:
            issues.append("text_too_small")
            score -= 40
        elif visual_angle < 0.5:
            issues.append("text_borderline_size")
            score -= 15

    # Contrast assessment
    if wcag_ratio is not None:
        if wcag_ratio < 3.0:
            issues.append("contrast_fail")
            score -= 35
        elif wcag_ratio < 4.5:
            issues.append("contrast_marginal")
            score -= 15

    if michelson is not None and michelson < 0.3:
        issues.append("low_pixel_contrast")
        score -= 10

    score = max(0, score)

    if score >= 75:
        text_behavior = "readable"
    elif score >= 45:
        text_behavior = "challenging"
    else:
        text_behavior = "unreadable"

    return {
        "text_behavior_score": score,
        "text_behavior_pred":  text_behavior,
        "text_behavior_issues": issues,
    }


def predict_human_behavior(visual_angle: float | None, logmar: float | None,
                           wcag_ratio: float | None,
                           michelson: float | None) -> dict:
    """
    Predict human reading difficulty, adjusted for human visual acuity.

    The acuity weight fine-tunes the raw text difficulty score:
    a text that is borderline for perfect vision becomes much harder
    for someone with reduced acuity (higher logMAR).

    difficulty_score = (100 - text_behavior_score) * acuity_weight
    Clamped to [0, 100]. Higher = harder for a human to read.
    """
    text_pred = predict_text_behavior(visual_angle, logmar, wcag_ratio, michelson)
    raw_difficulty = 100 - text_pred["text_behavior_score"]
    acuity_w = _acuity_weight(logmar if logmar is not None else 0.0)
    difficulty = min(100, round(raw_difficulty * acuity_w, 1))

    if difficulty < 25:
        human_pred = "easy"
    elif difficulty < 55:
        human_pred = "moderate_effort"
    elif difficulty < 80:
        human_pred = "difficult"
    else:
        human_pred = "very_difficult"

    return {
        "acuity_weight":        acuity_w,
        "human_difficulty_score": difficulty,
        "human_behavior_pred":  human_pred,
        **text_pred,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_all_metrics(
    response_id: str,
    cap_height_px: float | None,
    px_per_cm: float | None,
    text_rgb: list | None,
    bg_rgb: list | None,
    crops_dir: Path,
) -> dict:
    """
    Compute all Stage 4 metrics for a single submission.
    Returns flat dict for merging into the pipeline dataframe.
    """
    size = compute_size_metrics(cap_height_px, px_per_cm)
    crop_path = crops_dir / f"{response_id}_target.jpg"
    pixel_contrast = compute_pixel_contrast(crop_path)
    color_contrast = compute_color_contrast(text_rgb, bg_rgb)
    predictions = predict_human_behavior(
        visual_angle=size.get("visual_angle_deg"),
        logmar=size.get("logmar"),
        wcag_ratio=color_contrast.get("wcag_contrast_ratio"),
        michelson=pixel_contrast.get("michelson_contrast"),
    )

    return {
        "response_id": response_id,
        **size,
        **pixel_contrast,
        **color_contrast,
        **predictions,
    }
