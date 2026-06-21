"""
Stage 1 — CSV Ingestion & Data Cleaning
========================================
Loads the Qualtrics export, applies cleaning rules, and produces:
  - df_clean      : rows valid for image pipeline (has image + full consent)
  - df_survey_only: valid survey rows without usable images
  - failed_log    : all dropped rows with reason codes

Key update from Dr. He (Jun 15 email):
  One participant may submit multiple samples. Use the anonymized email
  field as the participant identifier (more reliable than name).
  Multiple ResponseIds can share the same fake email → same participant.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

from config import (
    CSV_PATH, IMAGE_DIR, OUTPUTS_DIR, PII_COLUMNS,
    MIN_DURATION_SEC, PIPELINE_VERSION
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Column name aliases (Qualtrics exports have very long column names) ────────
COL_DISTRIBUTION   = "Distribution Channel"
COL_PROGRESS       = "Progress"
COL_DURATION       = "Duration (in seconds)"
COL_FINISHED       = "Finished"
COL_RESPONSE_ID    = "ResponseId"
COL_CONSENT        = "Please indicate whether you consent to participate in this study. - Selected Choice"
COL_FIRST_TIME     = "Is this your first time participating in the Wicked Words research study?"
COL_US_RESIDENT    = "Are you currently residing in the United States?"
COL_ENGLISH        = "Are you able to read and understand English?"
COL_AGE_18         = "Are you 18 years of age or older?"
COL_ENROLLED       = "Are you currently enrolled in the Wicked Problems, Wolfpack Solutions (WPWS) course at North Carolina State University?"
COL_IMG_ID         = "Photo Upload_Id"
COL_IMG_NAME       = "Photo Upload_Name"
COL_IMG_SIZE       = "Photo Upload_Size"
COL_IMG_TYPE       = "Photo Upload_Type"
COL_THEME          = "Theme Choice"
COL_LEGIBLE        = "Experience 1"
COL_EASY_READ      = "Experience 2"
COL_PROLONGED      = "Experience 3"
COL_VISION_CORR    = "Experience 4"
COL_CHALLENGE_TEXT = "Follow up 1"
COL_ENV_TEXT       = "Follow up 2"
COL_PERSONAL_TEXT  = "Follow up 3"
COL_ACTIONS        = "Follow up 3.1"
COL_FEEDBACK       = "Feedback Question"
COL_EMAIL_ID       = "Email Collection"
COL_PARTICIPANT_ID = "Name Collection"


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Qualtrics exports have 3 header rows:
      Row 0: column names
      Row 1: human-readable question text (skip)
      Row 2: ImportId metadata (skip)
    """
    df = pd.read_csv(csv_path, header=0, skiprows=[1, 2], low_memory=False)
    log.info(f"Loaded {len(df)} rows from {csv_path.name}")
    return df


def clean(df: pd.DataFrame, image_dir: Path) -> dict:
    """
    Apply all cleaning rules. Returns dict with:
      clean        : DataFrame ready for image pipeline
      survey_only  : DataFrame valid survey data, no usable image
      failed_log   : DataFrame of all dropped rows with reason
    """
    failed_rows = []
    df = df.copy()
    df["_flags"] = [[] for _ in range(len(df))]

    def drop(mask, reason):
        failed_rows.append(df[mask].assign(_drop_reason=reason))
        return df[~mask].copy()

    # ── 1. Drop preview / test submissions ─────────────────────────────────
    if COL_DISTRIBUTION in df.columns:
        df = drop(df[COL_DISTRIBUTION] == "preview", "preview_submission")
    log.info(f"After preview drop: {len(df)} rows")

    # ── 2. Drop incomplete progress ─────────────────────────────────────────
    if COL_PROGRESS in df.columns:
        df[COL_PROGRESS] = pd.to_numeric(df[COL_PROGRESS], errors="coerce")
        df = drop(df[COL_PROGRESS] < 100, "incomplete_progress")
    log.info(f"After progress drop: {len(df)} rows")

    # ── 3. Consent check ───────────────────────────────────────────────────
    if COL_CONSENT in df.columns:
        consented = df[COL_CONSENT].fillna("").str.startswith("Yes")
        df = drop(~consented, "no_consent")
    log.info(f"After consent drop: {len(df)} rows")

    # ── 4. Eligibility screens ──────────────────────────────────────────────
    elig_map = {
        COL_US_RESIDENT: "not_us_resident",
        COL_ENGLISH:     "cannot_read_english",
        COL_AGE_18:      "under_18",
        COL_ENROLLED:    "not_enrolled_wpws",
    }
    for col, reason in elig_map.items():
        if col in df.columns:
            df = drop(df[col].fillna("") != "Yes", reason)
    log.info(f"After eligibility drop: {len(df)} rows")

    # ── 5. Flag fast submissions ────────────────────────────────────────────
    if COL_DURATION in df.columns:
        df[COL_DURATION] = pd.to_numeric(df[COL_DURATION], errors="coerce")
        fast_mask = df[COL_DURATION] < MIN_DURATION_SEC
        df.loc[fast_mask, "_flags"] = df.loc[fast_mask, "_flags"].apply(
            lambda f: f + ["fast_submission"]
        )
        log.info(f"Flagged {fast_mask.sum()} fast submissions (< {MIN_DURATION_SEC}s)")

    # ── 6. Deduplicate by participant email ─────────────────────────────────
    #    Per Dr. He: one participant may submit multiple times.
    #    Email is the reliable cross-submission identifier.
    #    Keep all submissions (multiple photos per person is valid research data),
    #    but flag duplicates so analysts can study within-participant patterns.
    if COL_EMAIL_ID in df.columns:
        email_counts = df[COL_EMAIL_ID].value_counts()
        repeat_emails = email_counts[email_counts > 1].index
        repeat_mask = df[COL_EMAIL_ID].isin(repeat_emails)
        df.loc[repeat_mask, "_flags"] = df.loc[repeat_mask, "_flags"].apply(
            lambda f: f + ["repeat_participant"]
        )
        log.info(f"Flagged {repeat_mask.sum()} rows from repeat participants")

        # Assign participant index for grouping (anonymized)
        email_to_pid = {e: i for i, e in enumerate(df[COL_EMAIL_ID].dropna().unique())}
        df["participant_idx"] = df[COL_EMAIL_ID].map(email_to_pid)
        df["submission_number"] = df.groupby(COL_EMAIL_ID).cumcount() + 1

    # ── 7. Split: image pipeline vs survey-only ─────────────────────────────
    has_image = (
        df[COL_IMG_SIZE].notna() &
        (pd.to_numeric(df[COL_IMG_SIZE], errors="coerce") > 0)
        if COL_IMG_SIZE in df.columns else pd.Series(False, index=df.index)
    )

    # Also check image file actually exists on disk
    if COL_RESPONSE_ID in df.columns:
        def image_on_disk(row):
            rid = str(row.get(COL_RESPONSE_ID, ""))
            matches = list(image_dir.glob(f"{rid}_*"))
            return len(matches) > 0
        df["image_on_disk"] = df.apply(image_on_disk, axis=1)
    else:
        df["image_on_disk"] = False

    df_img = df[has_image & df["image_on_disk"]].copy()
    df_survey_only = df[~(has_image & df["image_on_disk"])].copy()
    log.info(f"Image pipeline rows: {len(df_img)}")
    log.info(f"Survey-only rows: {len(df_survey_only)}")

    # ── 8. Build failed log ─────────────────────────────────────────────────
    failed_df = pd.concat(failed_rows, ignore_index=True) if failed_rows else pd.DataFrame()

    # ── 9. Strip PII from research outputs ─────────────────────────────────
    pii_present = [c for c in PII_COLUMNS if c in df_img.columns]
    df_img_clean = df_img.drop(columns=pii_present, errors="ignore")
    df_survey_clean = df_survey_only.drop(columns=pii_present, errors="ignore")

    return {
        "clean": df_img_clean,
        "survey_only": df_survey_clean,
        "failed_log": failed_df,
        "raw_with_flags": df,  # full flagged df for internal use
    }


def get_image_path(response_id: str, image_dir: Path) -> Path | None:
    """Find image file for a given ResponseId (filename = ResponseId_originalname.ext)."""
    matches = list(image_dir.glob(f"{response_id}_*"))
    return matches[0] if matches else None


def save_outputs(results: dict, output_dir: Path):
    """Write cleaned datasets to outputs/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results["clean"].to_parquet(output_dir / "clean_image_rows.parquet", index=False)
    results["clean"].to_csv(output_dir / "clean_image_rows.csv", index=False)
    results["survey_only"].to_csv(output_dir / "survey_only_rows.csv", index=False)

    if not results["failed_log"].empty:
        results["failed_log"].to_csv(output_dir / f"failed_log_{ts}.csv", index=False)

    log.info(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    df_raw = load_csv(CSV_PATH)
    results = clean(df_raw, IMAGE_DIR)
    save_outputs(results, OUTPUTS_DIR)
    print(f"\n✓ Clean rows for image pipeline : {len(results['clean'])}")
    print(f"✓ Survey-only rows              : {len(results['survey_only'])}")
    print(f"✓ Dropped rows logged           : {len(results['failed_log'])}")
