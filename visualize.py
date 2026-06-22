"""
Wicked Words — Research Visualizations
========================================
Generates a multi-page PDF + individual PNGs covering:
  1. Submission funnel
  2. Theme distribution
  3. Reading experience ratings by theme
  4. Visual angle & logMAR distributions
  5. WCAG contrast ratio distribution + compliance
  6. Contrast vs. legibility
  7. Text size vs. readability
  8. Pixel-level contrast metrics
  9. Participant submission frequency
 10. Card detection / data quality breakdown
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV  = Path("outputs/results.csv")
FIGURES_DIR  = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
THEME_COLORS = {
    "Wonderful Wednesday": "#4CAF50",
    "Focus Friday":        "#2196F3",
    "Terrible Tuesday":    "#F44336",
    "Strain Saturday":     "#FF9800",
    "Other":               "#9E9E9E",
}
WCAG_GREEN  = "#4CAF50"
WCAG_RED    = "#F44336"
WCAG_YELLOW = "#FFC107"

def theme_short(t):
    if pd.isna(t): return "Other"
    for key in THEME_COLORS:
        if key in str(t): return key
    return "Other"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_CSV)
df["theme_short"] = df["survey_theme"].apply(theme_short)
df["legible"]     = df["legible"].map({True: True, False: False, "True": True, "False": False})
df["easy_to_read"]= df["easy_to_read"].map({True: True, False: False, "True": True, "False": False})
df["prolonged_ok"]= df["prolonged_ok"].map({True: True, False: False, "True": True, "False": False})

df_img = df[df["card_found"] == True].copy()
df_metrics = df[df["visual_angle_deg"].notna()].copy()

print(f"Total submissions : {len(df)}")
print(f"Card found        : {len(df_img)}")
print(f"Full metrics      : {len(df_metrics)}")

# ── Helper ────────────────────────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"  saved {name}.png")

# =============================================================================
# 1. SUBMISSION FUNNEL
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
stages = ["Raw\nsubmissions\n(88)", "Passed\nquality checks\n(51)", "Has\nphoto\n(38)",
          "Card\ndetected\n(24)", "Full\nmetrics\n(23)"]
values = [88, 51, 38, 24, 23]
colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5", "#1565C0"]
bars = ax.barh(stages[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.6)
for bar, val in zip(bars, values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontweight="bold")
ax.set_xlabel("Number of submissions")
ax.set_title("Data Pipeline Funnel", fontsize=14, fontweight="bold")
ax.set_xlim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "01_submission_funnel")
plt.close()

# =============================================================================
# 2. THEME DISTRIBUTION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

theme_counts = df["theme_short"].value_counts()
clrs = [THEME_COLORS.get(t, "#9E9E9E") for t in theme_counts.index]
axes[0].bar(theme_counts.index, theme_counts.values, color=clrs, edgecolor="white")
axes[0].set_title("Submissions by Theme", fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=20)
axes[0].spines[["top", "right"]].set_visible(False)
for i, v in enumerate(theme_counts.values):
    axes[0].text(i, v + 0.2, str(v), ha="center", fontweight="bold")

axes[1].pie(theme_counts.values, labels=theme_counts.index, colors=clrs,
            autopct="%1.0f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title("Theme Proportion", fontweight="bold")
fig.suptitle("Reading Theme Distribution", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "02_theme_distribution")
plt.close()

# =============================================================================
# 3. READING EXPERIENCE RATINGS BY THEME
# =============================================================================
exp_cols = {"Legible": "legible", "Easy to Read": "easy_to_read", "OK for Prolonged": "prolonged_ok"}
themes   = list(THEME_COLORS.keys())
x = np.arange(len(themes))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
for i, (label, col) in enumerate(exp_cols.items()):
    rates = []
    for t in themes:
        sub = df[df["theme_short"] == t][col]
        rates.append(sub.mean() * 100 if len(sub) > 0 else 0)
    ax.bar(x + i * width, rates, width, label=label,
           color=["#66BB6A", "#42A5F5", "#FFA726"][i], edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels(themes, rotation=15, ha="right")
ax.set_ylabel("% Positive responses")
ax.set_ylim(0, 115)
ax.set_title("Reading Experience Ratings by Theme", fontsize=14, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
for bar in ax.patches:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
save(fig, "03_experience_by_theme")
plt.close()

# =============================================================================
# 4. VISUAL ANGLE & LOGMAR DISTRIBUTIONS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df_metrics["visual_angle_deg"].dropna(), bins=12,
             color="#42A5F5", edgecolor="white")
axes[0].axvline(df_metrics["visual_angle_deg"].median(), color="red",
                linestyle="--", label=f'Median: {df_metrics["visual_angle_deg"].median():.2f}°')
axes[0].set_xlabel("Visual Angle (degrees)")
axes[0].set_ylabel("Count")
axes[0].set_title("Visual Angle Distribution", fontweight="bold")
axes[0].legend()
axes[0].spines[["top", "right"]].set_visible(False)

axes[1].hist(df_metrics["logmar"].dropna(), bins=12,
             color="#7E57C2", edgecolor="white")
axes[1].axvline(0.0,  color="green",  linestyle="--", linewidth=1.5, label="20/20 (0.0)")
axes[1].axvline(0.3,  color="orange", linestyle="--", linewidth=1.5, label="20/40 (0.3)")
axes[1].axvline(1.0,  color="red",    linestyle="--", linewidth=1.5, label="20/200 (1.0)")
axes[1].set_xlabel("logMAR")
axes[1].set_ylabel("Count")
axes[1].set_title("logMAR Distribution\n(lower = better acuity)", fontweight="bold")
axes[1].legend(fontsize=8)
axes[1].spines[["top", "right"]].set_visible(False)

fig.suptitle("Text Size Metrics", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "04_visual_angle_logmar")
plt.close()

# =============================================================================
# 5. WCAG CONTRAST RATIO DISTRIBUTION + COMPLIANCE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

wcag = df_metrics["wcag_contrast_ratio"].dropna()
n, bins, patches = axes[0].hist(wcag, bins=14, edgecolor="white")
for patch, left in zip(patches, bins):
    if left < 3.0:   patch.set_facecolor(WCAG_RED)
    elif left < 4.5: patch.set_facecolor(WCAG_YELLOW)
    else:            patch.set_facecolor(WCAG_GREEN)
axes[0].axvline(3.0, color="orange", linestyle="--", linewidth=1.5, label="AA Large (3:1)")
axes[0].axvline(4.5, color="green",  linestyle="--", linewidth=1.5, label="AA Normal (4.5:1)")
axes[0].axvline(7.0, color="blue",   linestyle="--", linewidth=1.5, label="AAA (7:1)")
axes[0].set_xlabel("WCAG Contrast Ratio")
axes[0].set_ylabel("Count")
axes[0].set_title("Contrast Ratio Distribution", fontweight="bold")
axes[0].legend(fontsize=8)
axes[0].spines[["top", "right"]].set_visible(False)

compliance = {
    "AA Large\n(≥3:1)":   df_metrics["wcag_aa_large"].sum(),
    "AA Normal\n(≥4.5:1)":df_metrics["wcag_aa_normal"].sum(),
    "AAA Normal\n(≥7:1)": df_metrics["wcag_aaa_normal"].sum(),
}
total = len(df_metrics["wcag_contrast_ratio"].dropna())
clrs2 = [WCAG_YELLOW, WCAG_GREEN, "#1565C0"]
bars = axes[1].bar(compliance.keys(), compliance.values(), color=clrs2, edgecolor="white")
axes[1].set_ylabel("Count")
axes[1].set_ylim(0, total + 3)
axes[1].set_title(f"WCAG Compliance (n={total})", fontweight="bold")
axes[1].spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, compliance.values()):
    pct = val / total * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val} ({pct:.0f}%)", ha="center", fontweight="bold")

fig.suptitle("Contrast & Accessibility", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "05_wcag_contrast")
plt.close()

# =============================================================================
# 6. CONTRAST RATIO vs. LEGIBILITY (scatter)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
plot_df = df_metrics[df_metrics["wcag_contrast_ratio"].notna() & df_metrics["legible"].notna()].copy()
for legible, color, label in [(True, WCAG_GREEN, "Legible"), (False, WCAG_RED, "Not legible")]:
    sub = plot_df[plot_df["legible"] == legible]
    ax.scatter(sub["wcag_contrast_ratio"], sub["visual_angle_deg"],
               c=color, label=label, alpha=0.75, s=80, edgecolors="white")
ax.axvline(4.5, color="gray", linestyle="--", linewidth=1, label="WCAG AA (4.5:1)")
ax.set_xlabel("WCAG Contrast Ratio")
ax.set_ylabel("Visual Angle (degrees)")
ax.set_title("Contrast Ratio vs. Visual Angle\ncolored by Legibility", fontsize=13, fontweight="bold")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "06_contrast_vs_legibility")
plt.close()

# =============================================================================
# 7. TEXT SIZE vs. READABILITY (box plots)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in [
    (axes[0], "easy_to_read",  "Easy to Read"),
    (axes[1], "prolonged_ok",  "OK for Prolonged Reading"),
]:
    groups = {}
    for val, label in [(True, "Yes"), (False, "No")]:
        data = df_metrics[df_metrics[col] == val]["visual_angle_deg"].dropna()
        if len(data): groups[label] = data
    if groups:
        bp = ax.boxplot(groups.values(), tick_labels=groups.keys(), patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], [WCAG_GREEN, WCAG_RED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_ylabel("Visual Angle (degrees)")
    ax.set_title(title, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Text Size vs. Readability Ratings", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "07_textsize_vs_readability")
plt.close()

# =============================================================================
# 8. PIXEL-LEVEL CONTRAST METRICS (grouped bars)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
contrast_cols = {"Michelson": "michelson_contrast", "RMS": "rms_contrast", "MAD": "mad_contrast"}
means  = [df_metrics[c].mean()  for c in contrast_cols.values()]
stds   = [df_metrics[c].std()   for c in contrast_cols.values()]
colors = ["#42A5F5", "#66BB6A", "#FFA726"]
bars = ax.bar(contrast_cols.keys(), means, yerr=stds, color=colors,
              edgecolor="white", capsize=6)
ax.set_ylabel("Contrast Value (0–1)")
ax.set_ylim(0, 1.15)
ax.set_title("Pixel-Level Contrast Metrics\n(mean ± std)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
            f"{mean:.3f}", ha="center", fontweight="bold")
fig.tight_layout()
save(fig, "08_pixel_contrast_metrics")
plt.close()

# =============================================================================
# 9. PARTICIPANT SUBMISSION FREQUENCY
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sub_counts = df.groupby("participant_idx")["submission_number"].max().value_counts().sort_index()
axes[0].bar(sub_counts.index.astype(int), sub_counts.values,
            color="#7E57C2", edgecolor="white")
axes[0].set_xlabel("Number of submissions per participant")
axes[0].set_ylabel("Number of participants")
axes[0].set_title("Submission Frequency per Participant", fontweight="bold")
axes[0].spines[["top", "right"]].set_visible(False)
for i, v in zip(sub_counts.index, sub_counts.values):
    axes[0].text(i, v + 0.1, str(v), ha="center", fontweight="bold")

sub_trend = df.groupby("submission_number")[["legible","easy_to_read","prolonged_ok"]].mean() * 100
sub_trend = sub_trend[sub_trend.index <= 5]
for col, color, label in [("legible", "#66BB6A", "Legible"),
                           ("easy_to_read", "#42A5F5", "Easy to Read"),
                           ("prolonged_ok", "#FFA726", "Prolonged OK")]:
    if col in sub_trend.columns:
        axes[1].plot(sub_trend.index, sub_trend[col], marker="o", color=color, label=label)
axes[1].set_xlabel("Submission number")
axes[1].set_ylabel("% Positive")
axes[1].set_ylim(0, 110)
axes[1].set_title("Experience Ratings by Submission Order", fontweight="bold")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)

fig.suptitle("Participant Engagement", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "09_participant_frequency")
plt.close()

# =============================================================================
# 10. DATA QUALITY BREAKDOWN
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
quality = {
    "Privacy\nflagged":    int(df["privacy_flag"].sum()),
    "Card not\nfound":     int((df["card_found"] == False).sum()),
    "Card\nfound":         int((df["card_found"] == True).sum()),
    "Target\nfound":       int(df["target_found"].sum()),
    "Full\nmetrics":       int(df["visual_angle_deg"].notna().sum()),
}
colors = [WCAG_RED, WCAG_YELLOW, WCAG_GREEN, "#42A5F5", "#1565C0"]
bars = ax.bar(quality.keys(), quality.values(), color=colors, edgecolor="white")
ax.set_ylabel("Count")
ax.set_title("Data Quality Breakdown (38 image submissions)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, quality.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(val), ha="center", fontweight="bold")
fig.tight_layout()
save(fig, "10_data_quality")
plt.close()

# =============================================================================
# EXPORT: combine all into one PDF
# =============================================================================
png_files = sorted(FIGURES_DIR.glob("*.png"))
with PdfPages(FIGURES_DIR / "wicked_words_research_report.pdf") as pdf:
    for png in png_files:
        img = plt.imread(png)
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.imshow(img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

print(f"\n✓ All figures saved to {FIGURES_DIR}")
print(f"✓ Combined PDF: {FIGURES_DIR}/wicked_words_research_report.pdf")
