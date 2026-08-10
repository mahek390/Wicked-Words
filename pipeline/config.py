"""
Wicked Words Pipeline — Configuration
======================================
Central place for all paths, thresholds, and model settings.
Adjust GEMINI_API_KEY via environment variable before running.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT              = Path(__file__).parent.parent
DATA_DIR          = ROOT / "Data"
IMAGE_DIR         = DATA_DIR / "Photo Upload"
CROPS_DIR         = ROOT / "outputs" / "crops"
OUTPUTS_DIR       = ROOT / "outputs"
CSV_PATH          = DATA_DIR / "survey.csv"
REFCARD_PDF       = DATA_DIR / "WPWS-Wicked Words Postcard - shared with design team.pdf"

# De-identified copies — these are what get sent to AI services
DEIDENT_DIR       = OUTPUTS_DIR / "deidentified"
DEIDENT_IMAGE_DIR = DEIDENT_DIR / "images"
DEIDENT_CSV_PATH  = DEIDENT_DIR / "survey_deidentified.csv"

# ── Local model ──────────────────────────────────────────────────────────────
# Florence-2 runs locally on Apple Silicon — no API key needed
FLORENCE_MODEL_ID = "microsoft/Florence-2-large"

# ── Google Drive source ──────────────────────────────────────────────────────
# Photos can be synced from a shared Drive folder into IMAGE_DIR instead of
# being copied in by hand. Auth is via a service account — share the folder
# with that account's client_email as a Viewer. See pipeline/drive_fetch.py.
DRIVE_FOLDER_ID           = os.getenv("DRIVE_FOLDER_ID", "")
DRIVE_SERVICE_ACCOUNT_FILE = ROOT / os.getenv("DRIVE_SERVICE_ACCOUNT_FILE", "service-account-key.json")

# ── Calibration ───────────────────────────────────────────────────────────────
CALIB_BAR_CM        = 8.0    # physical width of the calibration bar on the reference card
VIEWING_DISTANCE_CM = 40.0   # standard viewing distance specified on the card (16 in)

# ── Quality thresholds ────────────────────────────────────────────────────────
MIN_PX_PER_CM        = 10.0   # below this → card too far / too small to trust
MAX_PX_PER_CM        = 500.0  # above this → card unusually large / macro shot
MIN_DURATION_SEC     = 60     # submissions faster than this are flagged
MAX_CARD_ANGLE_DEG   = 30     # tilt beyond this → flag for perspective correction
MIN_BAR_WIDTH_PX     = 20     # calibration bar narrower than this → unreliable

# ── De-identification ─────────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.5   # DNN face detector confidence cutoff
FACE_BLUR_SCALE           = 99    # Gaussian blur kernel size for faces

# ── Privacy — PII columns stripped before any output or AI call ───────────────
PII_COLUMNS = [
    "IPAddress",
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "LocationLatitude",
    "LocationLongitude",
    "Name Collection",
    "Email Collection",
    "ExternalReference",
]

# ── Output ────────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "0.2.0"
