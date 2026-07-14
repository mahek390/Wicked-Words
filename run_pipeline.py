"""
Wicked Words Pipeline — Main Runner
=====================================
Stage 0 : De-identification  — blur faces/plates in images, strip PII from CSV
Stage 1 : Data cleaning      — filter invalid submissions, assign participant IDs
Stage 2 : Ref card detection — Gemini detects calibration card, computes px/cm
Stage 3 : Target text        — Gemini extracts text region, colors, cap height
Stage 4 : Metrics            — visual angle, logMAR, contrast, readability predictions
Stage 5 : Fusion + output    — merge all results, write CSV / parquet

IMPORTANT: Stages 2 and 3 send images to Gemini.
           They ONLY receive de-identified images from Stage 0 output.
           Original images never leave the local machine.

Outputs (in outputs/):
    deidentified/images/          — blurred copies sent to AI
    deidentified/survey_deidentified.csv
    results.csv / results.parquet — full merged analytics
    participant_summary.csv
    flagged_for_review.csv
    failed_log_<ts>.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "pipeline"))

import pipeline.stage0_deidentify as s0
import pipeline.stage1_cleaning   as s1
import pipeline.stage2_refcard    as s2
import pipeline.stage3_target     as s3
import pipeline.stage4_metrics    as s4
import pipeline.stage5_fusion     as s5
from config import IMAGE_DIR, CSV_PATH, CROPS_DIR, OUTPUTS_DIR, DEIDENT_IMAGE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("runner")


def run(csv_path: Path, image_dir: Path, limit: int | None = None):
    log.info("=" * 60)
    log.info("Wicked Words Pipeline starting")
    log.info("=" * 60)

    # ── Stage 1: Clean CSV ────────────────────────────────────────────────────
    log.info("\n── Stage 1: Data cleaning ──")
    df_raw = s1.load_csv(csv_path)
    cleaning = s1.clean(df_raw, image_dir)
    s1.save_outputs(cleaning, OUTPUTS_DIR)

    df_clean = cleaning["clean"]
    if limit:
        df_clean = df_clean.head(limit)
        log.info(f"Running on first {limit} rows (--limit flag)")
    log.info(f"Processing {len(df_clean)} image rows")

    # ── Stage 0: De-identification ────────────────────────────────────────────
    # Must run BEFORE any image is sent to an AI service.
    # Uses original image_dir as source; writes blurred copies to deidentified/.
    log.info("\n── Stage 0: De-identification (privacy firewall) ──")
    deident = s0.run(df_raw, image_dir)
    deident_image_dir = deident["deident_image_dir"]
    log.info(f"  AI will use images from: {deident_image_dir}")

    # ── Stages 2–4: Per-image processing (using de-identified images) ─────────
    stage2_results = []
    stage3_results = []
    stage4_results = []

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing images"):
        rid = str(row.get(s1.COL_RESPONSE_ID, ""))

        # Always resolve image path from de-identified folder
        deident_path = s1.get_image_path(rid, deident_image_dir)
        if deident_path is None:
            log.warning(f"{rid}: de-identified image not found — skipping")
            continue

        # Stage 2: Reference card (uses de-identified image)
        r2 = s2.process_image(rid, deident_path, CROPS_DIR)
        stage2_results.append(r2)

        if r2.get("privacy_flag"):
            log.warning(f"{rid}: residual privacy flag after de-identification — skipping stages 3+4")
            continue
        if not r2.get("card_found"):
            log.info(f"{rid}: card not found — skipping stages 3+4")
            continue

        # Stage 3: Target text (uses de-identified image)
        r3 = s3.process_image(rid, deident_path, CROPS_DIR, card_bbox_frac=r2.get("card_bbox"))
        stage3_results.append(r3)

        if not r3.get("target_found"):
            log.info(f"{rid}: target not found — skipping stage 4")
            continue

        # Stage 4: Metrics + predictions (local computation, no AI)
        r4 = s4.compute_all_metrics(
            response_id=rid,
            cap_height_px=r3.get("cap_height_px"),
            px_per_cm=r2.get("px_per_cm"),
            text_rgb=r3.get("text_color_rgb"),
            bg_rgb=r3.get("background_color_rgb"),
            crops_dir=CROPS_DIR,
        )
        stage4_results.append(r4)

    log.info(f"\nStage summary:")
    log.info(f"  Stage 2 processed : {len(stage2_results)}")
    log.info(f"  Stage 3 processed : {len(stage3_results)}")
    log.info(f"  Stage 4 processed : {len(stage4_results)}")

    # ── Stage 5: Merge + save ─────────────────────────────────────────────────
    log.info("\n── Stage 5: Survey fusion + output ──")
    df_merged = s5.merge(df_clean, stage2_results, stage3_results, stage4_results)
    s5.save_outputs(df_merged, OUTPUTS_DIR)

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info(f"Results in: {OUTPUTS_DIR}")
    log.info("=" * 60)
    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wicked Words image analysis pipeline")
    parser.add_argument("--csv",    type=Path, default=CSV_PATH,  help="Path to Qualtrics CSV export")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR, help="Directory of submitted images")
    parser.add_argument("--limit",  type=int,  default=None,      help="Process only first N image rows (for testing)")
    args = parser.parse_args()
    run(args.csv, args.images, args.limit)
