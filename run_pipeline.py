"""
Wicked Words Pipeline — Main Runner
=====================================
Orchestrates all 5 stages for a given data directory.

Usage:
    python run_pipeline.py --csv data/survey.csv --images data/images/ [--limit 10]

Environment:
    export GEMINI_API_KEY="AIzaSyAPgrMIsXV9nc6QajoTiYRnH1yi0lzOCog"

Outputs (in outputs/):
    results.parquet           — full merged analytics dataset
    results.csv               — same, CSV format
    participant_summary.csv   — aggregated per-participant metrics
    flagged_for_review.csv    — rows needing human RA review
    failed_log_<ts>.csv       — dropped rows with reason codes
    clean_image_rows.csv      — Stage 1 output (reference)
"""

import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent / "pipeline"))

import os
import sys

# Track down the absolute path to your pipeline folder
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, "pipeline")
sys.path.append(pipeline_path)

# Now Python can see it as a direct, local file
import pipeline.stage1_cleaning as s1
import pipeline.stage2_refcard as s2
import pipeline.stage3_target   as s3
import pipeline.stage4_metrics  as s4
import pipeline.stage5_fusion   as s5
from config import IMAGE_DIR, CSV_PATH, CROPS_DIR, OUTPUTS_DIR

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

    # ── Stages 2–4: Per-image processing ─────────────────────────────────────
    stage2_results = []
    stage3_results = []
    stage4_results = []

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing images"):
        rid = str(row.get(s1.COL_RESPONSE_ID, ""))
        img_path = s1.get_image_path(rid, image_dir)

        if img_path is None:
            log.warning(f"{rid}: image file not found on disk")
            continue

        # Stage 2: Reference card
        log.debug(f"{rid}: Stage 2 — ref card detection")
        r2 = s2.process_image(rid, img_path, CROPS_DIR)
        stage2_results.append(r2)

        # Skip if privacy flagged or card not found
        if r2.get("privacy_flag"):
            log.warning(f"{rid}: privacy flag — skipping stages 3+4")
            continue
        if not r2.get("card_found"):
            log.info(f"{rid}: card not found — skipping stages 3+4")
            continue

        # Stage 3: Target text
        log.debug(f"{rid}: Stage 3 — target text extraction")
        r3 = s3.process_image(rid, img_path, CROPS_DIR, card_bbox_frac=r2.get("card_bbox"))
        stage3_results.append(r3)

        if not r3.get("target_found"):
            log.info(f"{rid}: target not found — skipping stage 4")
            continue

        # Stage 4: Metrics
        log.debug(f"{rid}: Stage 4 — metrics")
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
    parser.add_argument("--csv",    type=Path, default=CSV_PATH,   help="Path to Qualtrics CSV export")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR,  help="Directory of submitted images")
    parser.add_argument("--limit",  type=int,  default=None,       help="Process only first N image rows (for testing)")
    args = parser.parse_args()
    run(args.csv, args.images, args.limit)
