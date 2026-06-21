"""
Stage 3 — Target Text Detection & Extraction
=============================================
Identifies the "wicked" text in the scene background (the text the
participant found difficult to read), crops it out, and extracts:
  - bounding box (fractional)
  - OCR'd text content (via Gemini)
  - theme classification
  - estimated font height in pixels
  - background / text color estimates

The reference card is explicitly excluded — Gemini is told to ignore it
and focus only on the environmental text behind the participant's hand.
"""

import json
import time
import logging
from pathlib import Path
from PIL import Image

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_RETRIES, GEMINI_RETRY_DELAY,
)
from stage2_refcard import _call_gemini, bbox_crop, refine_mask_with_sam
import numpy as np

log = logging.getLogger(__name__)


# ── Gemini prompt ─────────────────────────────────────────────────────────────

TARGET_SYSTEM = (
    "You are a vision analysis assistant for the Wicked Words citizen science project. "
    "Participants photograph text they find difficult to read in everyday life while "
    "holding a Wicked Words reference card in front of the scene. "
    "Your job is to find and analyze the BACKGROUND TEXT — the text the participant "
    "found difficult — NOT the reference card itself. "
    "Return ONLY valid JSON. No markdown, no explanation."
)

TARGET_PROMPT = """This photo shows a person holding a Wicked Words reference card in front of some text.

IGNORE the reference card entirely.

Find the TARGET TEXT — the text in the scene background that the participant
found difficult to read. It could be: a sign, menu, food label, newspaper,
poster, package, digital screen, book, or any other everyday text.

Return JSON:
{
  "target_found": true | false,
  "target_bbox": [x1, y1, x2, y2] | null,
  "text_content": "<exact text if readable, truncated to 300 chars>" | null,
  "text_readable": true | false,
  "text_theme": "signage" | "packaging" | "print_media" | "digital_screen" | "handwritten" | "other",
  "estimated_cap_height_px": <height of a capital letter in pixels> | null,
  "background_luminance": "light" | "dark" | "mixed",
  "text_luminance": "light" | "dark" | "mixed",
  "text_color_rgb": [R, G, B] | null,
  "background_color_rgb": [R, G, B] | null,
  "scene_description": "<one sentence describing where/what the text is>",
  "confidence": "high" | "medium" | "low"
}

All bounding box coordinates are FRACTIONS of image width/height (0.0 to 1.0).
estimated_cap_height_px: height of a typical capital letter in the target text, in pixels.
"""


# ── Main entry point ──────────────────────────────────────────────────────────

def process_image(
    response_id: str,
    image_path: Path,
    crops_dir: Path,
    card_bbox_frac: list[float] | None = None,
) -> dict:
    """
    Run Stage 3 for a single image.
    card_bbox_frac: from Stage 2 — used to mask out card region if needed.
    Returns result dict for pipeline dataframe.
    """
    result = {
        "response_id": response_id,
        "target_found": False,
        "target_bbox": None,
        "text_content": None,
        "text_readable": None,
        "text_theme": None,
        "cap_height_px": None,
        "background_luminance": None,
        "text_luminance": None,
        "text_color_rgb": None,
        "background_color_rgb": None,
        "scene_description": None,
        "target_confidence": None,
        "stage3_flags": [],
    }

    gemini_out = _call_gemini(image_path, TARGET_PROMPT, TARGET_SYSTEM)
    if gemini_out is None:
        result["stage3_flags"].append("gemini_failed")
        return result

    result["target_found"]       = gemini_out.get("target_found", False)
    result["target_bbox"]        = gemini_out.get("target_bbox")
    result["text_content"]       = gemini_out.get("text_content")
    result["text_readable"]      = gemini_out.get("text_readable", False)
    result["text_theme"]         = gemini_out.get("text_theme")
    result["cap_height_px"]      = gemini_out.get("estimated_cap_height_px")
    result["background_luminance"] = gemini_out.get("background_luminance")
    result["text_luminance"]     = gemini_out.get("text_luminance")
    result["text_color_rgb"]     = gemini_out.get("text_color_rgb")
    result["background_color_rgb"] = gemini_out.get("background_color_rgb")
    result["scene_description"]  = gemini_out.get("scene_description")
    result["target_confidence"]  = gemini_out.get("confidence", "low")

    if not result["target_found"]:
        result["stage3_flags"].append("target_not_found")
        return result

    if result["target_confidence"] == "low":
        result["stage3_flags"].append("low_confidence")

    if not result["text_readable"]:
        result["stage3_flags"].append("text_unreadable")
        # Still valid — participant intentionally photographed hard-to-read text

    # ── Save target crop ──────────────────────────────────────────────────────
    if result["target_bbox"]:
        img = Image.open(image_path)
        try:
            # Attempt SAM refinement (returns None if SAM unavailable)
            img_np = np.array(img)
            sam_mask = refine_mask_with_sam(img_np, result["target_bbox"])

            if sam_mask is not None:
                # Apply mask: zero out non-target pixels
                masked = img_np.copy()
                masked[~sam_mask] = 255
                target_img = Image.fromarray(masked)
            else:
                target_img = bbox_crop(img, result["target_bbox"])

            crops_dir.mkdir(parents=True, exist_ok=True)
            target_img.save(crops_dir / f"{response_id}_target.jpg")
        except Exception as e:
            log.warning(f"{response_id}: crop save failed — {e}")
            result["stage3_flags"].append("crop_save_failed")

    return result
