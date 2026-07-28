"""
Stage 5 — Survey Fusion & Final Output
========================================
Merges image-derived metrics (Stages 2–4) with Qualtrics survey
responses into analytics-ready output files.

Also computes participant-level aggregates using the email-based
participant identifier per Dr. He's guidance (Jun 15 email):
  "one participant could submit multiple samples... email should be
   a more reliable identifier."
"""

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import OUTPUTS_DIR, PIPELINE_VERSION, PII_COLUMNS
from stage1_cleaning import (
    COL_RESPONSE_ID, COL_THEME, COL_LEGIBLE, COL_EASY_READ,
    COL_PROLONGED, COL_VISION_CORR, COL_CHALLENGE_TEXT,
    COL_ENV_TEXT, COL_PERSONAL_TEXT, COL_ACTIONS, COL_EMAIL_ID,
    COL_FEEDBACK,
)

log = logging.getLogger(__name__)

"""
pipeline/stage5_fusion.py
=========================
Annotates bounding boxes on image and appends an extended bottom metrics banner.
"""

import cv2
import numpy as np

def generate_annotated_dashboard(
    image_np: np.ndarray,
    card_info: dict,
    text_metrics: list[dict],
    survey_data: dict
) -> np.ndarray:
    """
    Creates an annotated image with a bottom banner displaying survey data and metrics.
    """
    img = image_np.copy()
    H, W = img.shape[:2]

    # 1. Annotate Reference Card (Green Box + Confidence)
    if card_info.get("card_found") and card_info.get("bbox"):
        cx1, cy1, cx2, cy2 = card_info["bbox"]
        conf = card_info.get("confidence", 0.0)
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        cv2.putText(
            img, f"Ref Card (Conf: {conf:.2f})", (cx1, max(cy1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # 2. Annotate Text Boxes (Cyan Boxes + Labels)
    for t in text_metrics:
        tx1, ty1, tx2, ty2 = t["bbox"]
        label = t["label"]
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
        cv2.putText(
            img, label, (tx1, max(ty1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1
        )

    # 3. Create Extended Bottom Banner
    banner_height = 220 + (len(text_metrics) * 22)
    banner = np.ones((banner_height, W, 3), dtype=np.uint8) * 245  # Light gray background

    # Divider line
    cv2.line(banner, (0, 5), (W, 5), (50, 50, 50), 2)

    # Render Survey Details
    y_offset = 30
    cv2.putText(banner, "--- SURVEY RESPONSES ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 150), 2)
    y_offset += 25
    
    theme = survey_data.get("theme", "N/A")
    yn1 = survey_data.get("readable_yn", "N/A")
    yn2 = survey_data.get("lighting_yn", "N/A")
    yn3 = survey_data.get("clear_bg_yn", "N/A")
    open_ans = survey_data.get("open_ended_response", "N/A")

    cv2.putText(banner, f"Theme: {theme}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 20
    cv2.putText(banner, f"Yes/No Qs -> Readable: {yn1} | Good Lighting: {yn2} | Clear BG: {yn3}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 20
    cv2.putText(banner, f"Open Answer: {open_ans[:80]}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Render Calculated Metrics
    y_offset += 30
    cv2.putText(banner, "--- CALCULATED METRICS ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 0, 0), 2)
    y_offset += 25

    ref_h_px = card_info.get("card_height_px", "N/A")
    cv2.putText(banner, f"Ref Card Height (px): {ref_h_px} | Conf: {card_info.get('confidence')}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 22

    for t in text_metrics:
        lbl = t["label"]
        v_ang = t.get("visual_angle_deg", "N/A")
        rel_ratio = t.get("relative_visual_angle_ratio", "N/A")
        contrast = t.get("michelson_contrast", "N/A")

        text_str = f"{lbl} -> Visual Angle: {v_ang} deg | Rel Angle Ratio: {rel_ratio} | Contrast: {contrast}"
        cv2.putText(banner, text_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        y_offset += 20

    # Stack original image and bottom banner together
    combined_img = np.vstack([img, banner])
    return combined_img

# ── Survey column rename map ──────────────────────────────────────────────────
# Maps long Qualtrics column names → short analysis-friendly names

SURVEY_RENAME = {
    COL_RESPONSE_ID:    "response_id",
    COL_THEME:          "survey_theme",
    COL_LEGIBLE:        "legible",
    COL_EASY_READ:      "easy_to_read",
    COL_PROLONGED:      "prolonged_ok",
    COL_VISION_CORR:    "vision_correction",
    COL_CHALLENGE_TEXT: "challenge_text",
    COL_ENV_TEXT:       "environment_text",
    COL_PERSONAL_TEXT:  "personal_factor_text",
    COL_ACTIONS:        "actions_taken",
    COL_FEEDBACK:       "feedback",
    COL_EMAIL_ID:       "participant_email_anon",
}
# Timestamp-like columns to exclude from the research spreadsheet
_TIMESTAMP_COLS = {
    "StartDate", "EndDate", "RecordedDate", "processed_at",
    "Duration (in seconds)",
}

# Internal pipeline columns not relevant to researchers
_INTERNAL_COLS = {
    "pipeline_version", "_flags", "all_flags",
    "image_on_disk", "stage2_flags", "stage3_flags",
    "text_behavior_issues", "Distribution Channel", "UserLanguage",
    "Status", "Finished", "Progress",
}

# The 3 image-derived DVs that go into the spreadsheet
_IMAGE_DVS = [
    "visual_angle_calib_multiples",
    "rms_contrast",
    "michelson_contrast",
]


def export_research_spreadsheet(df_merged: pd.DataFrame, output_dir: Path):
    """
    Produce outputs/research_data.xlsx with one row per submission:
      - P-xxx participant token (first column)
      - All non-PII, non-timestamp survey columns
      - Image DVs: visual_angle_calib_multiples, rms_contrast, michelson_contrast
    """
    drop = set(PII_COLUMNS) | _TIMESTAMP_COLS | _INTERNAL_COLS

    # Build participant token map from anonymised email
    if "participant_email_anon" in df_merged.columns:
        unique_emails = df_merged["participant_email_anon"].dropna().unique()
        token_map = {e: f"P-{i+1:03d}" for i, e in enumerate(sorted(unique_emails))}
        df_merged = df_merged.copy()
        df_merged["participant_id"] = df_merged["participant_email_anon"].map(token_map)
    else:
        df_merged = df_merged.copy()
        df_merged["participant_id"] = [
            f"P-{i+1:03d}" for i in range(len(df_merged))
        ]

    # Survey columns: everything not in drop set and not an image DV
    survey_cols = [
        c for c in df_merged.columns
        if c not in drop
        and c not in _IMAGE_DVS
        and c != "participant_id"
        and c != "participant_email_anon"
    ]

    # Image DVs — only include those that exist
    dv_cols = [c for c in _IMAGE_DVS if c in df_merged.columns]

    final_cols = ["participant_id"] + survey_cols + dv_cols
    df_out = df_merged[[c for c in final_cols if c in df_merged.columns]]

    out_path = output_dir / "research_data.xlsx"
    df_out.to_excel(out_path, index=False, sheet_name="Wicked Words Data")
    log.info(f"Research spreadsheet saved → {out_path}  ({len(df_out)} rows, {len(df_out.columns)} cols)")
    return out_path

BOOL_COLS = ["legible", "easy_to_read", "prolonged_ok"]


def _recode_bools(df: pd.DataFrame) -> pd.DataFrame:
    """Recode Yes/No survey fields to boolean."""
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": True, "No": False})
    return df


def merge(
    df_survey: pd.DataFrame,
    stage2_results: list[dict],
    stage3_results: list[dict],
    stage4_results: list[dict],
) -> pd.DataFrame:
    """
    Join all stage outputs to the cleaned survey dataframe.
    Returns the merged analytics dataframe.
    """
    # Rename survey columns
    rename_map = {k: v for k, v in SURVEY_RENAME.items() if k in df_survey.columns}
    df = df_survey.rename(columns=rename_map).copy()
    df = _recode_bools(df)

    # Merge image stage results
    for results, prefix in [
        (stage2_results, "s2"),
        (stage3_results, "s3"),
        (stage4_results, "s4"),
    ]:
        if results:
            df_stage = pd.DataFrame(results)
            if "response_id" in df_stage.columns:
                df = df.merge(df_stage, on="response_id", how="left")

    # Consolidate flags from all stages
    flag_cols = [c for c in df.columns if c.endswith("_flags")]
    def _merge_flags(row):
        seen = set()
        for col in flag_cols:
            val = row[col]
            if isinstance(val, list):
                seen.update(f for f in val if f)
        return list(seen)
    df["all_flags"] = df[flag_cols].apply(_merge_flags, axis=1)

    # Pipeline metadata
    df["pipeline_version"] = PIPELINE_VERSION
    df["processed_at"] = datetime.utcnow().isoformat()

    return df


def participant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate submission-level data to participant level.
    Uses anonymized email as the grouping key (per Dr. He).
    """
    if "participant_email_anon" not in df.columns:
        log.warning("No participant email column — skipping participant summary")
        return pd.DataFrame()

    agg = df.groupby("participant_email_anon").agg(
        n_submissions=("response_id", "count"),
        n_with_image=("card_found", "sum"),
        mean_visual_angle=("visual_angle_deg", "mean"),
        mean_logmar=("logmar", "mean"),
        mean_michelson=("michelson_contrast", "mean"),
        mean_rms=("rms_contrast", "mean"),
        mean_mad=("mad_contrast", "mean"),
        pct_legible=("legible", "mean"),
        pct_easy_read=("easy_to_read", "mean"),
        themes=("survey_theme", lambda x: list(x.dropna().unique())),
    ).reset_index()

    agg["mean_visual_angle"] = agg["mean_visual_angle"].round(4)
    agg["mean_logmar"] = agg["mean_logmar"].round(3)
    agg["mean_michelson"] = agg["mean_michelson"].round(4)
    agg["mean_mad"] = agg["mean_mad"].round(4)
    agg["pct_legible"] = agg["pct_legible"].round(3)

    return agg


def save_outputs(df_merged: pd.DataFrame, output_dir: Path):
    """Write final outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Primary research file (no PII — already stripped in Stage 1)
    df_merged.to_parquet(output_dir / "results.parquet", index=False)
    df_merged.to_csv(output_dir / "results.csv", index=False)

    # Participant-level summary
    summary = participant_summary(df_merged)
    if not summary.empty:
        summary.to_csv(output_dir / "participant_summary.csv", index=False)

    # Flagged rows for RA review
    needs_review = df_merged[
        df_merged["all_flags"].apply(lambda f: any(
            x in f for x in ["privacy_review_required", "low_confidence", "scale_error"]
        ))
    ]
    if not needs_review.empty:
        needs_review.to_csv(output_dir / "flagged_for_review.csv", index=False)

    log.info(f"Saved {len(df_merged)} merged rows to {output_dir}")
    log.info(f"  → {len(needs_review)} flagged for manual review")

    # Research spreadsheet for analysis
    export_research_spreadsheet(df_merged, output_dir)
