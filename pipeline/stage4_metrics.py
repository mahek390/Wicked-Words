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
  • cap_height_cm      : physical cap height in centimetres

  Contrast (pixel-level, from target crop)
  ─────────────────────────────────────────
  • michelson_contrast : (Lmax - Lmin) / (Lmax + Lmin) — global luminance range
  • rms_contrast       : std(luminance) / 128 — overall variability
  • mad_contrast       : median absolute deviation — robust to glare/outliers
  • mean_luminance     : mean pixel luminance (0–255)

  Predictions
  ────────────
  • text_behavior_score / text_behavior_pred  : optical readability of the scene
  • human_difficulty_score / human_behavior_pred : acuity-adjusted reading difficulty
"""

import math
import logging
import numpy as np
from pathlib import Path
from PIL import Image

from config import VIEWING_DISTANCE_CM

log = logging.getLogger(__name__)


# ── Visual angle ──────────────────────────────────────────────────────────────

def visual_angle_deg(cap_height_px: float, px_per_cm: float,
                     viewing_dist_cm: float = VIEWING_DISTANCE_CM) -> float:
    height_cm = cap_height_px / px_per_cm
    angle_rad = 2 * math.atan(height_cm / (2 * viewing_dist_cm))
    return math.degrees(angle_rad)


def logmar_from_deg(va_deg: float) -> float:
    va_arcmin = va_deg * 60.0
    if va_arcmin <= 0:
        return float("nan")
    return round(math.log10(va_arcmin), 3)


def snellen_from_logmar(logmar: float) -> str:
    if math.isnan(logmar):
        return "N/A"
    denom = 20 * (10 ** logmar)
    standards = [10, 12, 15, 20, 25, 30, 40, 50, 63, 80, 100, 125, 160, 200, 250, 320, 400]
    return f"20/{min(standards, key=lambda s: abs(s - denom))}"


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


# ── Pixel-level contrast (from target crop) ───────────────────────────────────

def compute_pixel_contrast(target_crop_path: Path) -> dict:
    empty = {"michelson_contrast": None, "rms_contrast": None,
             "mad_contrast": None, "mean_luminance": None}
    if not target_crop_path.exists():
        return empty

    img = Image.open(target_crop_path).convert("L")
    arr = np.array(img, dtype=float)
    if arr.size == 0:
        return empty

    L_max, L_min = arr.max(), arr.min()
    return {
        "michelson_contrast": round((L_max - L_min) / (L_max + L_min + 1e-6), 4),
        "rms_contrast":       round(arr.std() / 128.0, 4),
        "mad_contrast":       round(float(np.median(np.abs(arr - np.median(arr)))) / 128.0, 4),
        "mean_luminance":     round(float(arr.mean()), 2),
    }


# ── Readability prediction ────────────────────────────────────────────────────

ACUITY_BREAKPOINTS = [
    (0.0, 1.0),
    (0.3, 1.3),
    (0.5, 1.6),
    (1.0, 2.5),
    (2.0, 4.0),
]


def _acuity_weight(logmar: float) -> float:
    """Interpolate difficulty weight from logMAR (clinical acuity standard)."""
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
                          michelson: float | None, rms: float | None) -> dict:
    """
    Predict whether the text environment is objectively readable.
    Based purely on physical/optical metrics — no human input.

    Thresholds from vision science literature:
      Visual angle < 0.1°   → below acuity limit for most people
      Visual angle 0.1–0.5° → borderline, requires effort
      Michelson < 0.3       → low global luminance contrast
      RMS < 0.15            → low overall contrast variability
    """
    issues = []
    score = 100

    if visual_angle is not None:
        if visual_angle < 0.1:
            issues.append("text_too_small")
            score -= 40
        elif visual_angle < 0.5:
            issues.append("text_borderline_size")
            score -= 15

    if michelson is not None and michelson < 0.3:
        issues.append("low_michelson_contrast")
        score -= 25

    if rms is not None and rms < 0.15:
        issues.append("low_rms_contrast")
        score -= 20

    score = max(0, score)
    if score >= 75:   pred = "readable"
    elif score >= 45: pred = "challenging"
    else:             pred = "unreadable"

    return {"text_behavior_score": score, "text_behavior_pred": pred,
            "text_behavior_issues": issues}


def predict_human_behavior(visual_angle: float | None, logmar: float | None,
                           michelson: float | None, rms: float | None) -> dict:
    """
    Predict human reading difficulty, fine-tuned for human visual acuity.
    difficulty = (100 - text_score) * acuity_weight, clamped to [0, 100].
    Higher score = harder for a human to read.
    """
    text_pred  = predict_text_behavior(visual_angle, logmar, michelson, rms)
    acuity_w   = _acuity_weight(logmar if logmar is not None else 0.0)
    difficulty = min(100, round((100 - text_pred["text_behavior_score"]) * acuity_w, 1))

    if difficulty < 25:   human_pred = "easy"
    elif difficulty < 55: human_pred = "moderate_effort"
    elif difficulty < 80: human_pred = "difficult"
    else:                 human_pred = "very_difficult"

    return {
        "acuity_weight":          acuity_w,
        "human_difficulty_score": difficulty,
        "human_behavior_pred":    human_pred,
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
    size           = compute_size_metrics(cap_height_px, px_per_cm)
    pixel_contrast = compute_pixel_contrast(crops_dir / f"{response_id}_target.jpg")
    color_contrast = compute_color_contrast(text_rgb, bg_rgb)
    predictions    = predict_human_behavior(
        visual_angle=size.get("visual_angle_deg"),
        logmar=size.get("logmar"),
        michelson=pixel_contrast.get("michelson_contrast"),
        rms=pixel_contrast.get("rms_contrast"),
    )
    return {
        "response_id": response_id,
        **size,
        **pixel_contrast,
        **predictions,
    }
