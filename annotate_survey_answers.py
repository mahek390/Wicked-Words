"""
Annotate Survey Answers Only
===============================
Standalone script — no card detection, no text detection, no visual-angle
or contrast analysis. For every submission photo, appends a banner of just
its survey answers underneath the full, unmodified image.

Column names are resolved against the CSV's own header at runtime rather
than hardcoded blind: anything in FIELD_SPECS that isn't found is logged
and skipped instead of silently mismatched. Run with --list-columns to
print every column name pandas actually loaded from the CSV, to confirm
FIELD_SPECS matches your export exactly (Qualtrics exports occasionally
shift a column name between waves).

Banner labels use the actual question wording, read directly from the
CSV's own second row (Qualtrics stores the full question text there) —
not a guessed paraphrase. The short strings in FIELD_SPECS are only a
fallback for columns where that row is blank.

The two "Follow up 3" questions share an identical header in the raw CSV;
pandas auto-renames the second one to "Follow up 3.1" on load (Stage 1 of
the main pipeline already relies on this same behavior).

Output: outputs/survey_annotated/<response_id>_survey.jpg
"""

import os
import re
import sys
import logging
import argparse
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "pipeline"))

import pipeline.stage1_cleaning as s1
from config import IMAGE_DIR, CSV_PATH, OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("annotate_survey")

OUTPUT_DIR = OUTPUTS_DIR / "survey_annotated"

# (fallback label, exact CSV column name). Order here is render order.
# The fallback label is only used if the CSV's question-text row is blank
# for that column — see load_question_text().
FIELD_SPECS = [
    ("Theme",                        "Theme Choice"),
    ("Experience 1",                 "Experience 1"),
    ("Experience 2",                 "Experience 2"),
    ("Experience 3",                 "Experience 3"),
    ("Experience 4",                 "Experience 4"),
    ("Experience 4 (write-in)",      "Experience 4_4_TEXT"),
    ("Follow up 1",                  "Follow up 1"),
    ("Follow up 2",                  "Follow up 2"),
    ("Follow up 3 (eyes?)",          "Follow up 3"),
    ("Follow up 3 (actions taken?)", "Follow up 3.1"),
    ("Feedback",                     "Feedback Question"),
]

LINE_HEIGHT = 22
WRAP_WIDTH = 100
LABEL_MAX_LEN = 90
BANNER_BG = 245
HEADER_COLOR = (150, 0, 0)   # BGR
BODY_COLOR = (20, 20, 20)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_question_text(csv_path: Path) -> dict[str, str]:
    """
    Qualtrics' second CSV row holds the full question wording for every
    column (row 0 is the short field code, row 2 is the ImportId JSON).
    Reading it directly means banner labels use the survey's own words
    instead of a hand-guessed paraphrase, and it dedupes duplicate headers
    ("Follow up 3" -> "Follow up 3.1") the exact same way s1.load_csv does,
    so the keys line up with the data columns.
    """
    header_and_text = pd.read_csv(csv_path, header=0, nrows=1, low_memory=False)
    if header_and_text.empty:
        return {}
    row = header_and_text.iloc[0]
    text_by_col = {}
    for col in header_and_text.columns:
        raw = row[col]
        if pd.isna(raw):
            continue
        text = _HTML_TAG_RE.sub("", str(raw)).strip()
        if text:
            text_by_col[col] = text[:LABEL_MAX_LEN] + ("…" if len(text) > LABEL_MAX_LEN else "")
    return text_by_col


def resolve_columns(df_columns, question_text: dict[str, str] | None = None) -> list[tuple[str, str | None]]:
    """
    Match each wanted field against the CSV's actual columns by exact name,
    preferring the real question wording (from load_question_text) as the
    display label and falling back to the short FIELD_SPECS label only if
    that row was blank for this column. Anything not found in the data at
    all is logged and kept as (label, None) so the banner still renders
    "N/A" for it instead of the whole run failing.
    """
    question_text = question_text or {}
    available = set(df_columns)
    resolved = []
    for fallback_label, col in FIELD_SPECS:
        label = question_text.get(col, fallback_label)
        if col in available:
            resolved.append((label, col))
        else:
            log.warning(f"Column not found in CSV, will show N/A: {col!r}")
            resolved.append((label, None))
    return resolved


def _clean(value):
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = str(value).strip()
    return text if text else None


def draw_survey_banner(image_path: Path, survey_row: dict, fields: list[tuple[str, str | None]]) -> np.ndarray:
    """Full original image, unmodified, with a survey-answers banner appended below."""
    img = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    W = img_bgr.shape[1]

    lines = [("SURVEY RESPONSES", HEADER_COLOR, 0.6, 2)]
    for label, col in fields:
        value = _clean(survey_row.get(col)) if col else None
        text = f"{label}: {value if value is not None else 'N/A'}"
        wrapped_lines = textwrap.wrap(text, WRAP_WIDTH) or [text]
        for wrapped in wrapped_lines:
            lines.append((wrapped, BODY_COLOR, 0.46, 1))

    banner_h = 30 + len(lines) * LINE_HEIGHT
    banner = np.full((banner_h, W, 3), BANNER_BG, dtype=np.uint8)
    cv2.line(banner, (0, 2), (W, 2), (60, 60, 60), 2)

    y = 26
    for text, color, scale, thickness in lines:
        cv2.putText(banner, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += LINE_HEIGHT

    return np.vstack([img_bgr, banner])


def run(csv_path: Path, image_dir: Path, limit: int | None = None):
    log.info(f"Loading {csv_path} ...")
    df = s1.load_csv(csv_path)
    question_text = load_question_text(csv_path)
    fields = resolve_columns(df.columns, question_text)
    for label, col in fields:
        log.info(f"  banner label for {col!r}: {label!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = df if limit is None else df.head(limit)
    n_written, n_no_image = 0, 0

    for _, row in rows.iterrows():
        rid = str(row.get(s1.COL_RESPONSE_ID, "")).strip()
        if not rid:
            continue

        img_path = s1.get_image_path(rid, image_dir)
        if img_path is None:
            n_no_image += 1
            continue

        try:
            combined = draw_survey_banner(img_path, row.to_dict(), fields)
        except Exception as e:
            log.warning(f"{rid}: failed to annotate — {e}")
            continue

        out_path = OUTPUT_DIR / f"{rid}_survey.jpg"
        cv2.imwrite(str(out_path), combined)
        n_written += 1

    log.info(f"Done. {n_written} annotated images written to {OUTPUT_DIR}")
    log.info(f"{n_no_image} survey rows had no matching photo and were skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate every submission photo with its survey answers only — no detection or analysis."
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to Qualtrics CSV export")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR, help="Directory of submitted images")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for testing)")
    parser.add_argument(
        "--list-columns", action="store_true",
        help="Print every column name (and its question text, if any) from the CSV, then exit.",
    )
    args = parser.parse_args()

    if args.list_columns:
        df = s1.load_csv(args.csv)
        question_text = load_question_text(args.csv)
        for col in df.columns:
            text = question_text.get(col)
            print(f"{col}" + (f"  ->  {text}" if text else ""))
        sys.exit(0)

    run(args.csv, args.images, args.limit)