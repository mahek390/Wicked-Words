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

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / "Data"
IMAGE_DIR    = DATA_DIR / "Photo Upload"
CROPS_DIR    = ROOT / "outputs" / "crops"
OUTPUTS_DIR  = ROOT / "outputs"
CSV_PATH     = DATA_DIR / "survey.csv"
REFCARD_PDF  = DATA_DIR / "WPWS-Wicked Words Postcard - shared with design team.pdf"

# ── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL     = "gemini-2.5-pro"
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 2.0   # seconds, doubles on each retry

# ── Calibration ──────────────────────────────────────────────────────────────
CALIB_BAR_CM        = 8.0    # physical width of the calibration bar on the reference card
VIEWING_DISTANCE_CM = 40.0   # standard viewing distance specified on the card (16 in)

# ── Quality thresholds ───────────────────────────────────────────────────────
MIN_PX_PER_CM        = 10.0   # below this → card too far / too small to trust
MAX_PX_PER_CM        = 500.0  # above this → card unusually large / macro shot
MIN_DURATION_SEC     = 60     # submissions faster than this are flagged
MAX_CARD_ANGLE_DEG   = 30     # tilt beyond this → flag for perspective correction
MIN_BAR_WIDTH_PX     = 20     # calibration bar narrower than this → unreliable

# ── Privacy ──────────────────────────────────────────────────────────────────
# Columns stripped before writing shared research files
PII_COLUMNS = [
    "IP Address",
    "Recipient Last Name",
    "Recipient First Name",
    "Recipient Email",
    "Location Latitude",
    "Location Longitude",
    "Please enter your full name.",
    "Please enter your secondary identifier below.\n\n\tPlease enter your email address below.",
]

# ── Output ───────────────────────────────────────────────────────────────────
PIPELINE_VERSION = "0.1.0"
