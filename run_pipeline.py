"""
Wicked Words Pipeline — Main Runner
=====================================
Stage 0 : De-identification  — blur faces/plates in images, strip PII from CSV (Skipped for raw processing)
Stage 1 : Data cleaning      — filter invalid submissions, assign participant IDs
Stage 2 : Ref card detection — Local model detects calibration card, computes px/cm, confidence
Stage 3 : Multi-target text  — Extracts all text blocks (text1, text2...), calculates relative 
                               ratios to bar, coordinates [x1, y1, x2, y2], and visual angles
Stage 4 : Metrics            — Computes logMAR, legibility/readability predictions
Stage 5 : Fusion + output    — Merges survey responses (Theme, Y/N, open-ended) with 
                               calculated image/text metrics into CSV / Parquet
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "pipeline"))

import pipeline.stage0_deidentify as s0
import pipeline.stage1_cleaning   as s1
import pipeline.stage2_refcard    as s2
import pipeline.stage3_target     as s3
import pipeline.stage4_metrics    as s4
import pipeline.stage5_fusion     as s5
# ── IMPORT ANNOTATOR MODULE HERE ──────────────────────────────────────────────
import pipeline.stage2_stage3_annotator as annotator

from config import IMAGE_DIR, CSV_PATH, CROPS_DIR, OUTPUTS_DIR, DEIDENT_IMAGE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("runner")

# Output directory for annotated full images with survey dashboard banner
ANNOTATED_DIR = OUTPUTS_DIR / "annotated_images"


def run(csv_path: Path, image_dir: Path, limit: int | None = None):
    log.info("=" * 60)
    log.info("Wicked Words Pipeline starting")
    log.info("=" * 60)

    # ── Stage 1: Data cleaning ────────────────────────────────────────────────
    log.info("\n── Stage 1: Data cleaning ──")
    df_raw = s1.load_csv(csv_path)
    cleaning = s1.clean(df_raw, image_dir)
    s1.save_outputs(cleaning, OUTPUTS_DIR)

    df_clean = cleaning["clean"]
    if limit:
        df_clean = df_clean.head(limit)
        log.info(f"Running on first {limit} rows (--limit flag)")
    log.info(f"Processing {len(df_clean)} image rows")

    # Raw image directory (skipping Stage 0 de-identification)
    deident_image_dir = IMAGE_DIR
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Stages 2–4: Per-image processing ──────────────────────────────────────
    stage2_results = []
    stage3_results = []
    stage4_results = []

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing images"):
        rid = str(row.get(s1.COL_RESPONSE_ID, ""))

        # Resolve image path
        deident_path = s1.get_image_path(rid, deident_image_dir)
        if deident_path is None:
            log.warning(f"{rid}: image not found — skipping")
            continue

        # Stage 2: Reference card detection & calibration
        r2 = s2.process_image(rid, deident_path, CROPS_DIR)
        stage2_results.append(r2)

        if r2.get("privacy_flag"):
            log.warning(f"{rid}: residual privacy flag — skipping stages 3+4")
            continue

        # Extract calibration metrics if card was found
        px_per_cm = r2.get("px_per_cm") if r2.get("card_found") else None
        card_bbox = r2.get("card_bbox") if r2.get("card_found") else None

        # Resolve viewing distance
        raw_dist = row.get("distance_cm") or row.get("distance_ft")
        try:
            distance_cm = float(raw_dist) if raw_dist else None
            if distance_cm and "distance_ft" in row and row.get("distance_ft"):
                distance_cm = distance_cm * 30.48
        except (ValueError, TypeError):
            distance_cm = None

        # Stage 3: Multi-target text extraction
        r3 = s3.process_image(
            response_id=rid,
            image_path=deident_path,
            crops_dir=CROPS_DIR,
            px_per_cm=px_per_cm,
            distance_cm=distance_cm,
            card_bbox_frac=card_bbox,
        )
        stage3_results.append(r3)

        # ── ANNOTATION & DASHBOARD BANNER GENERATION ─────────────────────────
        try:
            raw_img = Image.open(deident_path).convert("RGB")
            W, H = raw_img.size

            # Format text blocks from Stage 3 for the annotator
            annot_text_blocks = []
            bar_px = r2.get("bar_width_px") or r2.get("corrected_bar_px") or 1.0

            for t in r3.get("text_blocks", []):
                # Ensure bounding box is in pixel tuple format [x1, y1, x2, y2]
                if "bbox_frac" in t:
                    bx1, by1, bx2, by2 = t["bbox_frac"]
                    bbox_px = (int(bx1 * W), int(by1 * H), int(bx2 * W), int(by2 * H))
                else:
                    bbox_px = tuple(t.get("bbox_px", (0, 0, 0, 0)))

                metrics = annotator.compute_visual_metrics(bbox_px, bar_height_px=bar_px, px_per_cm=px_per_cm)
                contrast = annotator.compute_contrast(s2.np.array(raw_img), bbox_px)

                annot_text_blocks.append({
                    "bbox_px": bbox_px,
                    "contrast": contrast,
                    **metrics
                })

            # Save full annotated image with dashboard banner (Uncropped)
            annotated_out_path = ANNOTATED_DIR / f"{rid}_annotated.jpg"
            annotator.draw_annotations_and_dashboard(
                image=raw_img,
                card_info=r2,
                text_regions=annot_text_blocks,
                survey_data=dict(row),  # Pass complete survey row dictionary
                output_path=annotated_out_path
            )
        except Exception as e:
            log.warning(f"{rid}: Image annotation failed — {e}")
        # ─────────────────────────────────────────────────────────────────────

        if not r3.get("target_found"):
            log.info(f"{rid}: target text not found — skipping stage 4")
            continue

        # Stage 4: Metrics computation per block
        r4 = s4.compute_all_metrics(
            response_id=rid,
            text_blocks=r3.get("text_blocks", []),
            px_per_cm=r2.get("px_per_cm"),
            calib_bar_px=r2.get("corrected_bar_px"),
            crops_dir=CROPS_DIR,
        )
        stage4_results.append(r4)

    log.info(f"\nStage summary:")
    log.info(f"  Stage 2 processed : {len(stage2_results)}")
    log.info(f"  Stage 3 processed : {len(stage3_results)}")
    log.info(f"  Stage 4 processed : {len(stage4_results)}")

    # ── Stage 5: Survey fusion + output ───────────────────────────────────────
    log.info("\n── Stage 5: Survey fusion + output ──")
    df_merged = s5.merge(df_clean, stage2_results, stage3_results, stage4_results)
    s5.save_outputs(df_merged, OUTPUTS_DIR)

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info(f"Results in: {OUTPUTS_DIR}")
    log.info(f"Annotated images saved to: {ANNOTATED_DIR}")
    log.info("=" * 60)
    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wicked Words image analysis pipeline")
    parser.add_argument("--csv",    type=Path, default=CSV_PATH,  help="Path to Qualtrics CSV export")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR, help="Directory of submitted images")
    parser.add_argument("--limit",  type=int,  default=None,      help="Process only first N image rows (for testing)")
    args = parser.parse_args()
    run(args.csv, args.images, args.limit)