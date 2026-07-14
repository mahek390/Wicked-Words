"""
Stage 2 — Reference Card Detection & Calibration
==================================================
For each image, uses Gemini to:
  1. Detect the Wicked Words postcard bounding box
  2. Locate the 8 cm calibration bar on the card
  3. Compute px/cm scale factor
  4. Flag quality issues (occluded, blurry, tilted)

The scale factor converts all subsequent pixel measurements to real-world cm,
enabling visual angle calculation and Snellen-equivalent acuity estimates.

Note: SAM integration is stubbed with a clear interface (need to swap in the real SAM predictor when GPU/checkpoint is available) Gemini bbox alone is
sufficient for an initial pipeline run.
"""

import json
import time
import math
import logging
import numpy as np
from pathlib import Path
from PIL import Image

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_RETRIES, GEMINI_RETRY_DELAY,
    CALIB_BAR_CM, MAX_CARD_ANGLE_DEG, MIN_PX_PER_CM, MAX_PX_PER_CM,
    MIN_BAR_WIDTH_PX,
)

log = logging.getLogger(__name__)
_client = None

def get_client():
    """Lazily initialize the Gemini client so modules importable without API key set."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


# ── Gemini prompt ─────────────────────────────────────────────────────────────

CARD_SYSTEM = (
    "You are a vision analysis assistant for the Wicked Words citizen science project. "
    "The reference card is a postcard (~15 cm × 10 cm) with: "
    "'WICKED WORDS' in large blue text on the left half; "
    "a Snellen-style eye chart on the right half; "
    "a solid black bar at top AND bottom labeled 'This bar should be 8 cm long.' "
    "Return ONLY valid JSON with no markdown, no explanation."
)

CARD_PROMPT = """Analyze this photo and return the following as JSON:

{
  "card_found": true | false,
  "card_bbox": [x1, y1, x2, y2] | null,
  "calibration_bar_bbox": [x1, y1, x2, y2] | null,
  "which_bar": "top" | "bottom" | null,
  "card_occluded": true | false,
  "card_blurry": true | false,
  "card_angle_deg": <estimated tilt in degrees, 0 = flat>,
  "confidence": "high" | "medium" | "low",
  "privacy_flag": true | false,
  "privacy_reason": "<face/plate/id detected>" | null
}

All bounding box coordinates are FRACTIONS of image width/height (0.0 to 1.0).
card_bbox: entire postcard region.
calibration_bar_bbox: just the solid black bar (top bar preferred; use bottom if top is occluded).
card_occluded: true if >30% of card is hidden by hand or object.
privacy_flag: true if any recognizable face, license plate, or ID document is visible ANYWHERE in the image.
"""


# ── Gemini call with retry ────────────────────────────────────────────────────

def _call_gemini(image_path: Path, prompt: str, system: str) -> dict | None:
    img = Image.open(image_path)
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = get_client().models.generate_content(
                model=GEMINI_MODEL,
                contents=[img, prompt],
                config=types.GenerateContentConfig(system_instruction=system),
            )
            raw = response.text.strip()
            # Strip markdown fences if model wraps output
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error on attempt {attempt+1}: {e}")
        except Exception as e:
            log.warning(f"Gemini error on attempt {attempt+1}: {e}")
            time.sleep(GEMINI_RETRY_DELAY * (2 ** attempt))
    return None


# ── Scale factor computation ──────────────────────────────────────────────────

def compute_scale_factor(
    img_width: int,
    img_height: int,
    bar_bbox_frac: list[float],
    card_angle_deg: float = 0.0,
) -> dict:
    """
    Convert the pixel width of the calibration bar (8 cm physical) to px/cm.

    If card is tilted, the bar's apparent pixel width is foreshortened.
    Corrects using: true_width = apparent_width / cos(angle).

    Returns dict with px_per_cm and tilt_corrected flag.
    """
    x1, y1, x2, y2 = bar_bbox_frac
    apparent_bar_px = (x2 - x1) * img_width

    if apparent_bar_px < MIN_BAR_WIDTH_PX:
        return {"px_per_cm": None, "error": f"bar too narrow ({apparent_bar_px:.1f}px)"}

    # Tilt correction
    angle_rad = math.radians(min(abs(card_angle_deg), 60))
    tilt_corrected = abs(card_angle_deg) > 5
    corrected_bar_px = apparent_bar_px / math.cos(angle_rad) if tilt_corrected else apparent_bar_px

    px_per_cm = corrected_bar_px / CALIB_BAR_CM

    if not (MIN_PX_PER_CM <= px_per_cm <= MAX_PX_PER_CM):
        return {
            "px_per_cm": px_per_cm,
            "error": f"px_per_cm={px_per_cm:.1f} outside plausible range [{MIN_PX_PER_CM}, {MAX_PX_PER_CM}]",
            "tilt_corrected": tilt_corrected,
        }

    return {
        "px_per_cm": round(px_per_cm, 2),
        "apparent_bar_px": round(apparent_bar_px, 1),
        "corrected_bar_px": round(corrected_bar_px, 1),
        "tilt_corrected": tilt_corrected,
        "error": None,
    }


# ── SAM stub 

def refine_mask_with_sam(image_np: np.ndarray, bbox_frac: list[float]) -> np.ndarray | None:
    """
    SAM integration stub.
    When GPU + SAM checkpoint (sam_vit_h.pth) are available, replace this body with:

        from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        predictor = SamPredictor(sam)
        H, W = image_np.shape[:2]
        x1,y1,x2,y2 = bbox_frac
        box = np.array([x1*W, y1*H, x2*W, y2*H])
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(box=box[None], multimask_output=False)
        return masks[0]   # boolean mask same shape as image

    Returns None when SAM is unavailable; callers fall back to bbox crop.
    """
    log.debug("SAM not available — using Gemini bbox directly")
    return None


# ── Crop helper ───────────────────────────────────────────────────────────────

def bbox_crop(img: Image.Image, bbox_frac: list[float]) -> Image.Image:
    W, H = img.size
    x1, y1, x2, y2 = bbox_frac
    return img.crop((int(x1*W), int(y1*H), int(x2*W), int(y2*H)))


# ── Main entry point ──────────────────────────────────────────────────────────

def process_image(response_id: str, image_path: Path, crops_dir: Path) -> dict:
    """
    Run Stage 2 for a single image.
    Returns a result dict to be merged into the pipeline dataframe row.
    """
    result = {
        "response_id": response_id,
        "card_found": False,
        "px_per_cm": None,
        "card_bbox": None,
        "calibration_bar_bbox": None,
        "card_angle_deg": None,
        "card_occluded": None,
        "card_blurry": None,
        "card_confidence": None,
        "privacy_flag": False,
        "privacy_reason": None,
        "scale_error": None,
        "stage2_flags": [],
    }

    # ── Call Gemini ──────────────────────────────────────────────────────────
    gemini_out = _call_gemini(image_path, CARD_PROMPT, CARD_SYSTEM)
    if gemini_out is None:
        result["scale_error"] = "gemini_failed"
        result["stage2_flags"].append("gemini_failed")
        return result

    result["card_found"]     = gemini_out.get("card_found", False)
    result["card_bbox"]      = gemini_out.get("card_bbox")
    result["calibration_bar_bbox"] = gemini_out.get("calibration_bar_bbox")
    result["card_angle_deg"] = gemini_out.get("card_angle_deg", 0.0)
    result["card_occluded"]  = gemini_out.get("card_occluded", False)
    result["card_blurry"]    = gemini_out.get("card_blurry", False)
    result["card_confidence"]= gemini_out.get("confidence", "low")
    result["privacy_flag"]   = gemini_out.get("privacy_flag", False)
    result["privacy_reason"] = gemini_out.get("privacy_reason")

    # ── Privacy gate ─────────────────────────────────────────────────────────
    if result["privacy_flag"]:
        result["stage2_flags"].append("privacy_review_required")
        log.warning(f"{response_id}: privacy flag — {result['privacy_reason']}")
        return result  # do not process further

    if not result["card_found"]:
        result["stage2_flags"].append("card_not_found")
        return result

    # Quality flags 
    if result["card_occluded"]:
        result["stage2_flags"].append("card_occluded")
    if result["card_blurry"]:
        result["stage2_flags"].append("card_blurry")
    if abs(result["card_angle_deg"] or 0) > MAX_CARD_ANGLE_DEG:
        result["stage2_flags"].append("high_tilt")
    if result["card_confidence"] == "low":
        result["stage2_flags"].append("low_confidence")

    # Compute scale factor 
    if result["calibration_bar_bbox"] is None:
        result["scale_error"] = "bar_not_found"
        result["stage2_flags"].append("bar_not_found")
        return result

    img = Image.open(image_path)
    W, H = img.size
    scale = compute_scale_factor(W, H, result["calibration_bar_bbox"], result["card_angle_deg"])
    result["px_per_cm"]   = scale.get("px_per_cm")
    result["scale_error"] = scale.get("error")
    if scale.get("error"):
        result["stage2_flags"].append("scale_error")

    # ── Save card crop ────────────────────────────────────────────────────────
    if result["card_bbox"]:
        crops_dir.mkdir(parents=True, exist_ok=True)
        card_crop = bbox_crop(img, result["card_bbox"])
        card_crop.save(crops_dir / f"{response_id}_card.jpg")

    return result
