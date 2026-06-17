"""
Wicked Words Data Pipeline
- Loads and cleans survey CSV
- Groups responses by participant (via email)
- Matches photos to responses via ResponseId
- Extracts image metadata (EXIF + basic stats)
- Outputs cleaned CSV and summary report
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


# ── Config ──────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["ResponseId", "Email Collection"]   # bare minimum to keep a row
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_survey(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def clean_survey(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows missing any required field
    df = df.dropna(subset=REQUIRED_FIELDS).copy()

    # Normalise email to lowercase for reliable grouping
    df["Email Collection"] = df["Email Collection"].str.lower().str.strip()

    # Drop rows where ResponseId is clearly a test or Qualtrics preview row
    df = df[~df["ResponseId"].astype(str).str.startswith("R_preview")]

    # Assign a stable participant ID based on email
    emails = df["Email Collection"].unique()
    email_to_pid = {e: f"P{i+1:04d}" for i, e in enumerate(sorted(emails))}
    df["ParticipantId"] = df["Email Collection"].map(email_to_pid)

    return df.reset_index(drop=True)


# ── Image utilities ───────────────────────────────────────────────────────────

def extract_exif(img: Image.Image) -> dict:
    raw = img._getexif()
    if not raw:
        return {}
    return {TAGS.get(k, k): v for k, v in raw.items() if isinstance(v, (str, int, float, bytes))}


def image_metrics(img_path: str) -> dict:
    metrics = {"image_path": img_path, "load_error": None}
    try:
        with Image.open(img_path) as img:
            img.verify()          # check integrity

        with Image.open(img_path) as img:
            metrics["width"], metrics["height"] = img.size
            metrics["mode"] = img.mode

            arr = np.array(img.convert("L"), dtype=np.float32)
            metrics["brightness_mean"] = float(arr.mean())
            metrics["brightness_std"] = float(arr.std())

            exif = extract_exif(img)
            metrics["ExposureTime"] = exif.get("ExposureTime")
            metrics["FNumber"] = exif.get("FNumber")
            metrics["ISOSpeedRatings"] = exif.get("ISOSpeedRatings")
            metrics["DateTimeOriginal"] = exif.get("DateTimeOriginal")
            metrics["GPSInfo"] = str(exif.get("GPSInfo", ""))

    except Exception as e:
        metrics["load_error"] = str(e)

    return metrics


def scan_photos(photo_dir: str) -> dict[str, dict]:
    """Return {ResponseId: metrics_dict} for every photo found."""
    results = {}
    for fname in os.listdir(photo_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in PHOTO_EXTENSIONS:
            continue
        full_path = os.path.join(photo_dir, fname)
        results[stem] = image_metrics(full_path)
    return results


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run(csv_path: str, photo_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load & clean survey
    print("Loading survey …")
    raw = load_survey(csv_path)
    print(f"  Raw rows: {len(raw)}")

    cleaned = clean_survey(raw)
    print(f"  After cleaning: {len(cleaned)} rows, {cleaned['ParticipantId'].nunique()} participants")

    # 2. Scan photos
    print("Scanning photos …")
    photo_metrics = scan_photos(photo_dir)
    print(f"  Found {len(photo_metrics)} photos")

    # 3. Merge on ResponseId
    photo_df = pd.DataFrame.from_dict(photo_metrics, orient="index")
    photo_df.index.name = "ResponseId"
    photo_df = photo_df.reset_index()

    merged = cleaned.merge(photo_df, on="ResponseId", how="left")

    # 4. Flags / QC column
    merged["has_photo"] = merged["image_path"].notna()
    merged["photo_load_error"] = merged["load_error"].notna() & merged["load_error"].ne("")

    # 5. Save outputs
    cleaned_path = os.path.join(output_dir, "survey_cleaned.csv")
    merged_path = os.path.join(output_dir, "survey_with_photos.csv")
    merged.to_csv(merged_path, index=False)
    cleaned.to_csv(cleaned_path, index=False)

    # 6. Summary report
    summary = {
        "total_raw_submissions": int(len(raw)),
        "valid_submissions": int(len(cleaned)),
        "unique_participants": int(cleaned["ParticipantId"].nunique()),
        "photos_found": int(len(photo_metrics)),
        "submissions_with_photo": int(merged["has_photo"].sum()),
        "photo_load_errors": int(merged["photo_load_error"].sum()),
        "submissions_per_participant": (
            cleaned.groupby("ParticipantId").size().describe().to_dict()
        ),
    }
    report_path = os.path.join(output_dir, "summary_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs written to: {output_dir}")
    print(json.dumps(summary, indent=2))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH   = "/Users/mahekiphone/Documents/GitHub/Wicked-Words/Data/survey.csv"
    PHOTO_DIR  = "/Users/mahekiphone/Documents/GitHub/Wicked-Words/Data/Photo Upload"
    OUTPUT_DIR = "/Users/mahekiphone/Documents/GitHub/Wicked-Words/output"

    run(CSV_PATH, PHOTO_DIR, OUTPUT_DIR)
