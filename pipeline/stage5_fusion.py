"""
Stage 5 — Survey Fusion & Final Output
========================================
Merges image-derived metrics (Stages 2–4) with Qualtrics survey
responses into analytics-ready output files.

Also generates annotated verification images with a bottom metrics banner
and computes participant-level aggregates using the email-based
participant identifier.
"""

import json
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image

from config import OUTPUTS_DIR, PIPELINE_VERSION, PII_COLUMNS
from stage1_cleaning import (
    COL_RESPONSE_ID, COL_THEME, COL_LEGIBLE, COL_EASY_READ,
    COL_PROLONGED, COL_VISION_CORR, COL_CHALLENGE_TEXT,
    COL_ENV_TEXT, COL_PERSONAL_TEXT, COL_ACTIONS, COL_EMAIL_ID,
    COL_FEEDBACK,
)

log = logging.getLogger(__name__)

# ── Survey column rename map ──────────────────────────────────────────────────
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

_TIMESTAMP_COLS = {
    "StartDate", "EndDate", "RecordedDate", "processed_at",
    "Duration (in seconds)",
}

_INTERNAL_COLS = {
    "pipeline_version", "_flags", "all_flags",
    "image_on_disk", "stage2_flags", "stage3_flags",
    "text_behavior_issues", "Distribution Channel", "UserLanguage",
    "Status", "Finished", "Progress",
}

_IMAGE_DVS = [
    "visual_angle_calib_multiples",
    "rms_contrast",
    "michelson_contrast",
]

BOOL_COLS = ["legible", "easy_to_read", "prolonged_ok"]


def generate_annotated_dashboard(
    image_np: np.ndarray,
    card_info: dict,
    text_metrics: list[dict],
    survey_data: dict
) -> np.ndarray:
    """
    Creates an annotated image with bounding boxes and a bottom banner displaying 
    survey data, detection status, and calculated visual metrics.
    """
    img = image_np.copy()
    H, W = img.shape[:2]

    # 1. Annotate Reference Card (Green Box + Confidence)
    card_found = card_info.get("card_found", False)
    if card_found and card_info.get("bbox"):
        cx1, cy1, cx2, cy2 = [int(v) for v in card_info["bbox"]]
        conf = card_info.get("confidence", 0.0)
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
        cv2.putText(
            img, f"Ref Card (Conf: {conf:.2f})", (cx1, max(cy1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # 2. Annotate Text Boxes (Cyan Boxes + Labels)
    for t in text_metrics:
        if "bbox" in t and t["bbox"]:
            tx1, ty1, tx2, ty2 = [int(v) for v in t["bbox"]]
            label = t.get("label", "Target Text")
            cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (255, 255, 0), 2)
            cv2.putText(
                img, label, (tx1, max(ty1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1
            )

    # 3. Create Extended Bottom Banner
    banner_height = max(240, 220 + (len(text_metrics) * 22))
    banner = np.ones((banner_height, W, 3), dtype=np.uint8) * 245  # Light gray background

    # Divider line
    cv2.line(banner, (0, 5), (W, 5), (50, 50, 50), 2)

    # Render Detection Status & Survey Details
    y_offset = 30
    cv2.putText(banner, "--- SURVEY & DETECTION STATUS ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 150), 2)
    y_offset += 25
    
    card_status = "DETECTED" if card_found else "NOT FOUND"
    text_status = "DETECTED" if len(text_metrics) > 0 else "NOT FOUND"
    cv2.putText(banner, f"Card Status: {card_status}  |  Target Text Status: {text_status}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 100, 0) if card_found else (0, 0, 200), 2)
    y_offset += 22

    theme = survey_data.get("survey_theme", survey_data.get("theme", "N/A"))
    yn1 = survey_data.get("legible", "N/A")
    yn2 = survey_data.get("easy_to_read", "N/A")
    yn3 = survey_data.get("prolonged_ok", "N/A")
    challenge_txt = str(survey_data.get("challenge_text", "N/A"))

    cv2.putText(banner, f"Theme: {theme}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 20
    cv2.putText(banner, f"Readable: {yn1} | Easy Read: {yn2} | Prolonged OK: {yn3}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 20
    cv2.putText(banner, f"Challenge Notes: {challenge_txt[:80]}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Render Calculated Metrics
    y_offset += 30
    cv2.putText(banner, "--- CALCULATED METRICS ---", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 0, 0), 2)
    y_offset += 25

    ref_h_px = card_info.get("card_height_px", "N/A")
    conf_val = card_info.get("confidence", "N/A")
    cv2.putText(banner, f"Ref Card Height: {ref_h_px} px | Conf: {conf_val}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 22

    if not text_metrics:
        cv2.putText(banner, "No text metrics calculated (Target text missing or card missing).", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
    else:
        for t in text_metrics:
            lbl = t.get("label", "Text")
            v_ang = t.get("visual_angle_deg", "N/A")
            rel_ratio = t.get("relative_visual_angle_ratio", "N/A")
            contrast = t.get("michelson_contrast", "N/A")

            text_str = f"{lbl} -> Visual Angle: {v_ang} deg | Rel Ratio: {rel_ratio} | Contrast: {contrast}"
            cv2.putText(banner, text_str, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
            y_offset += 20

    # Stack original image and bottom banner together
    combined_img = np.vstack([img, banner])
    return combined_img


def export_research_spreadsheet(df_merged: pd.DataFrame, output_dir: Path):
    """
    Produce outputs/research_data.xlsx with one row per submission.
    """
    drop = set(PII_COLUMNS) | _TIMESTAMP_COLS | _INTERNAL_COLS

    if "participant_email_anon" in df_merged.columns:
        unique_emails = df_merged["participant_email_anon"].dropna().unique()
        token_map = {e: f"P-{i+1:03d}" for i, e in enumerate(sorted(unique_emails))}
        df_merged = df_merged.copy()
        df_merged["participant_id"] = df_merged["participant_email_anon"].map(token_map)
    else:
        df_merged = df_merged.copy()
        df_merged["participant_id"] = [f"P-{i+1:03d}" for i in range(len(df_merged))]

    survey_cols = [
        c for c in df_merged.columns
        if c not in drop
        and c not in _IMAGE_DVS
        and c != "participant_id"
        and c != "participant_email_anon"
    ]

    dv_cols = [c for c in _IMAGE_DVS if c in df_merged.columns]

    final_cols = ["participant_id"] + survey_cols + dv_cols
    df_out = df_merged[[c for c in final_cols if c in df_merged.columns]]

    out_path = output_dir / "research_data.xlsx"
    df_out.to_excel(out_path, index=False, sheet_name="Wicked Words Data")
    log.info(f"Research spreadsheet saved → {out_path}  ({len(df_out)} rows, {len(df_out.columns)} cols)")
    return out_path


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
    """
    rename_map = {k: v for k, v in SURVEY_RENAME.items() if k in df_survey.columns}
    df = df_survey.rename(columns=rename_map).copy()
    df = _recode_bools(df)

    for results, prefix in [
        (stage2_results, "s2"),
        (stage3_results, "s3"),
        (stage4_results, "s4"),
    ]:
        if results:
            df_stage = pd.DataFrame(results)
            if "response_id" in df_stage.columns:
                df = df.merge(df_stage, on="response_id", how="left")

    flag_cols = [c for c in df.columns if c.endswith("_flags")]
    def _merge_flags(row):
        seen = set()
        for col in flag_cols:
            val = row[col]
            if isinstance(val, list):
                seen.update(f for f in val if f)
        return list(seen)
    
    if flag_cols:
        df["all_flags"] = df[flag_cols].apply(_merge_flags, axis=1)
    else:
        df["all_flags"] = [[] for _ in range(len(df))]

    df["pipeline_version"] = PIPELINE_VERSION
    df["processed_at"] = datetime.utcnow().isoformat()

    return df


def participant_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate submission-level data to participant level using anonymized email.
    """
    if "participant_email_anon" not in df.columns:
        log.warning("No participant email column — skipping participant summary")
        return pd.DataFrame()

    agg_dict = {
        "n_submissions": ("response_id", "count"),
        "n_with_image": ("card_found", "sum") if "card_found" in df.columns else ("response_id", "count"),
        "pct_legible": ("legible", "mean") if "legible" in df.columns else ("response_id", "count"),
        "pct_easy_read": ("easy_to_read", "mean") if "easy_to_read" in df.columns else ("response_id", "count"),
    }
    
    if "visual_angle_deg" in df.columns:
        agg_dict["mean_visual_angle"] = ("visual_angle_deg", "mean")
    if "michelson_contrast" in df.columns:
        agg_dict["mean_michelson"] = ("michelson_contrast", "mean")

    agg = df.groupby("participant_email_anon").agg(**agg_dict).reset_index()

    for num_col in ["mean_visual_angle", "mean_michelson", "pct_legible"]:
        if num_col in agg.columns:
            agg[num_col] = agg[num_col].round(4)

    return agg


def create_annotated_outputs(df_merged: pd.DataFrame, deidentified_images_dir: Path, output_dir: Path):
    """
    Generates annotated verification dashboards for each image in the dataset.
    """
    annotated_dir = output_dir / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df_merged.iterrows():
        rid = row.get("response_id")
        if not rid:
            continue

        img_path = deidentified_images_dir / f"{rid}_image.jpg"
        if not img_path.exists():
            continue

        # Load image via OpenCV
        img_np = cv2.imread(str(img_path))
        if img_np is None:
            continue

        # Build card info
        card_info = {
            "card_found": row.get("card_found", False),
            "bbox": row.get("card_bbox"),
            "confidence": row.get("card_confidence", 0.0),
            "card_height_px": row.get("card_height_px", "N/A"),
        }

        # Build text metrics list
        text_metrics = []
        if row.get("target_found", False) and row.get("target_bbox"):
            text_metrics.append({
                "label": "Target Text",
                "bbox": row.get("target_bbox"),
                "visual_angle_deg": row.get("visual_angle_deg", "N/A"),
                "relative_visual_angle_ratio": row.get("visual_angle_calib_multiples", "N/A"),
                "michelson_contrast": row.get("michelson_contrast", "N/A"),
            })

        # Build survey data dictionary
        survey_data = row.to_dict()

        # Generate dashboard image
        combined = generate_annotated_dashboard(img_np, card_info, text_metrics, survey_data)
        cv2.imwrite(str(annotated_dir / f"{rid}_annotated.jpg"), combined)

    log.info(f"Saved annotated dashboard images to {annotated_dir}")


def save_outputs(df_merged: pd.DataFrame, output_dir: Path, deidentified_images_dir: Path = None):
    """Write final outputs and annotated dashboards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean DataFrame of non-serializable objects (PIL Images, raw arrays) for Parquet/CSV export
    df_export = df_merged.copy()
    non_serializable = [c for c in df_export.columns if "img" in c.lower() and c not in ["image_on_disk"]]
    df_export = df_export.drop(columns=non_serializable, errors="ignore")

    # Primary research file
    df_export.to_parquet(output_dir / "results.parquet", index=False)
    df_export.to_csv(output_dir / "results.csv", index=False)

    # Generate visual dashboards if image directory is supplied
    if deidentified_images_dir and deidentified_images_dir.exists():
        create_annotated_outputs(df_merged, deidentified_images_dir, output_dir)

    # Participant-level summary
    summary = participant_summary(df_merged)
    if not summary.empty:
        summary.to_csv(output_dir / "participant_summary.csv", index=False)

    # Flagged rows for RA review
    if "all_flags" in df_merged.columns:
        needs_review = df_merged[
            df_merged["all_flags"].apply(lambda f: isinstance(f, list) and any(
                x in f for x in ["privacy_review_required", "low_confidence", "scale_error"]
            ))
        ]
        if not needs_review.empty:
            needs_review.to_csv(output_dir / "flagged_for_review.csv", index=False)
            log.info(f"  → {len(needs_review)} flagged for manual review")

    log.info(f"Saved {len(df_merged)} merged rows to {output_dir}")

    # Research spreadsheet for analysis
    export_research_spreadsheet(df_merged, output_dir)