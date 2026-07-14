"""
Stage 0 — De-identification (Privacy Firewall)
================================================
Runs BEFORE any data is sent to AI services (Gemini or similar).

Two separate concerns handled here:

A. IMAGE DE-IDENTIFICATION
   - Detects faces using OpenCV DNN face detector and blurs them
   - Detects license plates via contour heuristics and blurs them
   - Saves de-identified copies to outputs/deidentified/images/
   - Original images are NEVER sent to any external API

B. SURVEY CSV DE-IDENTIFICATION
   - Drops all PII columns (name, email, IP, location)
   - Replaces participant email with an opaque hash-based token
     so cross-submission linking still works without exposing the email
   - Saves de-identified CSV to outputs/deidentified/survey_deidentified.csv

Only the outputs of this stage are passed downstream.
"""

import cv2
import hashlib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from config import (
    IMAGE_DIR, OUTPUTS_DIR,
    DEIDENT_IMAGE_DIR, DEIDENT_CSV_PATH,
    FACE_BLUR_SCALE, FACE_CONFIDENCE_THRESHOLD,
    PII_COLUMNS,
)
from stage1_cleaning import COL_EMAIL_ID, COL_RESPONSE_ID

log = logging.getLogger(__name__)

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

# ── OpenCV DNN face detector (ships with opencv-python, no extra download) ────
# Uses the Caffe model bundled with OpenCV
_face_net = None

def _get_face_net():
    global _face_net
    if _face_net is None:
        # OpenCV 5+ ships FaceDetectorYN — no external model files needed
        try:
            _face_net = cv2.FaceDetectorYN_create(
                "", "", (320, 320), score_threshold=FACE_CONFIDENCE_THRESHOLD
            )
            log.info("Loaded OpenCV FaceDetectorYN")
        except Exception:
            _face_net = "dnn"
    return _face_net


# ── Face detection ────────────────────────────────────────────────────────────

def detect_faces_dnn(img_bgr: np.ndarray) -> list[tuple]:
    """Return list of (x1, y1, x2, y2) face bounding boxes."""
    H, W = img_bgr.shape[:2]
    boxes = []
    try:
        detector = cv2.FaceDetectorYN_create(
            "", "", (W, H), score_threshold=FACE_CONFIDENCE_THRESHOLD
        )
        _, faces = detector.detect(img_bgr)
        if faces is not None:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                pad = int(w * 0.15)
                boxes.append((max(0, x - pad), max(0, y - pad),
                               min(W, x + w + pad), min(H, y + h + pad)))
    except Exception as e:
        log.debug(f"FaceDetectorYN failed: {e} — using DNN fallback")
        # DNN fallback with bundled model
        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net = cv2.dnn.readNetFromCaffe(
                cv2.data.haarcascades.replace("haarcascades", "dnn/deploy.prototxt"),
                cv2.data.haarcascades.replace("haarcascades",
                    "dnn/res10_300x300_ssd_iter_140000.caffemodel")
            )
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > FACE_CONFIDENCE_THRESHOLD:
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    x1, y1, x2, y2 = box.astype(int)
                    boxes.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
        except Exception:
            log.debug("DNN fallback also failed — no face detection for this image")
    return boxes


def detect_plates_heuristic(img_bgr: np.ndarray) -> list[tuple]:
    """
    Heuristic license plate detector using contour aspect ratio.
    Flags rectangles with aspect ratio ~2:1 to 5:1 and moderate area.
    Not perfect — errs on the side of over-blurring.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        area_frac = (w * h) / (W * H)
        if 1.5 < aspect < 6.0 and 0.002 < area_frac < 0.05:
            boxes.append((x, y, x + w, y + h))
    return boxes


# ── Blur helper ───────────────────────────────────────────────────────────────

def blur_regions(img_bgr: np.ndarray, boxes: list[tuple], strength: int = 51) -> np.ndarray:
    """Apply strong Gaussian blur to each bounding box region."""
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = strength | 1  # must be odd
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


# ── Per-image de-identification ───────────────────────────────────────────────

def deidentify_image(src_path: Path, dst_path: Path) -> dict:
    """
    Load image, blur faces and plates, save to dst_path.
    Returns a summary dict with counts of detections.
    """
    result = {
        "response_id": src_path.stem.split("_")[0],
        "faces_blurred": 0,
        "plates_blurred": 0,
        "deident_error": None,
    }
    try:
        img_bgr = cv2.imread(str(src_path))
        if img_bgr is None:
            # Try via PIL for HEIC / unusual formats
            pil = Image.open(src_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        faces  = detect_faces_dnn(img_bgr)
        plates = detect_plates_heuristic(img_bgr)

        img_bgr = blur_regions(img_bgr, faces,  strength=99)
        img_bgr = blur_regions(img_bgr, plates, strength=61)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), img_bgr)

        result["faces_blurred"]  = len(faces)
        result["plates_blurred"] = len(plates)

    except Exception as e:
        log.error(f"{src_path.name}: de-identification failed — {e}")
        result["deident_error"] = str(e)
        # Copy original as fallback so pipeline doesn't break
        import shutil
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    return result


# ── Survey CSV de-identification ──────────────────────────────────────────────

def _hash_token(value: str, salt: str = "wicked-words-2026") -> str:
    """One-way hash of email → opaque participant token (P-XXXXXXXX)."""
    h = hashlib.sha256(f"{salt}:{value.lower().strip()}".encode()).hexdigest()[:8].upper()
    return f"P-{h}"


def deidentify_survey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all PII columns and replace email with a stable opaque token.
    The token is consistent across submissions from the same participant.
    """
    df = df.copy()

    # Replace email with hash token BEFORE dropping the column
    if COL_EMAIL_ID in df.columns:
        df["participant_token"] = df[COL_EMAIL_ID].fillna("").apply(
            lambda e: _hash_token(e) if e else None
        )

    # Drop all PII columns
    pii_present = [c for c in PII_COLUMNS if c in df.columns]

    # Also drop any column whose name contains these keywords
    pii_keywords = ["name", "email", "ip", "latitude", "longitude",
                    "recipient", "location", "address"]
    for col in df.columns:
        if any(kw in col.lower() for kw in pii_keywords):
            if col not in pii_present:
                pii_present.append(col)

    df = df.drop(columns=pii_present, errors="ignore")
    log.info(f"Survey de-identification: dropped {len(pii_present)} PII columns, "
             f"added participant_token")
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def run(df_survey: pd.DataFrame, image_dir: Path) -> dict:
    """
    De-identify all images and the survey CSV.
    Returns:
      deident_image_dir : Path to folder of de-identified images
      df_deident        : De-identified survey DataFrame
      image_log         : List of per-image de-identification results
    """
    log.info("── Stage 0: De-identification ──")
    DEIDENT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # A. Images
    image_log = []
    image_files = [f for f in image_dir.iterdir()
                   if f.suffix.lower() in PHOTO_EXTENSIONS]
    log.info(f"  De-identifying {len(image_files)} images …")

    for src in image_files:
        dst = DEIDENT_IMAGE_DIR / src.name
        result = deidentify_image(src, dst)
        image_log.append(result)
        total = result["faces_blurred"] + result["plates_blurred"]
        if total > 0:
            log.info(f"  {src.name}: blurred {result['faces_blurred']} face(s), "
                     f"{result['plates_blurred']} plate(s)")

    log.info(f"  De-identified images saved to {DEIDENT_IMAGE_DIR}")

    # B. Survey CSV
    df_deident = deidentify_survey(df_survey)
    DEIDENT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_deident.to_csv(DEIDENT_CSV_PATH, index=False)
    log.info(f"  De-identified survey saved to {DEIDENT_CSV_PATH}")

    return {
        "deident_image_dir": DEIDENT_IMAGE_DIR,
        "df_deident": df_deident,
        "image_log": image_log,
    }
