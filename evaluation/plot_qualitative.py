"""
Generate Figure 8 for Deep Scratch paper: qualitative prediction examples.

Layout: 3 columns × 2 rows
  - Top row: raw tri-axial accelerometer signals with ground-truth scratch
             regions shaded in light red
  - Bottom row: predicted scratch mask probabilities (pr2_prob) as a filled
                curve, with a dashed threshold line at 0.5

Columns: (a) True positive  (b) False negative  (c) False positive
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

# ---------- data ----------
samples = pd.read_csv("/sessions/bold-dreamy-dijkstra/mnt/uploads/qualitative_samples.csv")
summary = pd.read_csv("/sessions/bold-dreamy-dijkstra/mnt/uploads/qualitative_summary.csv")

# Pick one representative segment per panel type
# (a) TP — high confidence, good pr2 overlap
tp_seg = summary[summary.example_type == "true_positive"].sort_values(
    "pr1_prob", ascending=False).iloc[0]["segment"]
# (b) FN — most scratch steps missed
fn_seg = summary[summary.example_type == "false_negative"].sort_values(
    "gt_scratch_steps", ascending=False).iloc[0]["segment"]
# (c) FP — high confidence false alarm with many pr2 steps
fp_seg = summary[summary.example_type == "false_positive"].sort_values(
    "pr2_scratch_steps", ascending=False).iloc[0]["segment"]

panels = [
    ("(a) True Positive", tp_seg),
    ("(b) False Negative", fn_seg),
    ("(c) False Positive", fp_seg),
]

print("Selected segments:")
for label, seg in panels:
    row = summary[summary.segment == seg].iloc[0]
    print(f"  {label}: seq_len={row.seq_len}, gt_steps={row.gt_scratch_steps}, "
          f"pr1_prob={row.pr1_prob:.3f}, pr2_steps={row.pr2_scratch_steps}")

# ---------- style ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS_XYZ = ["#2166ac", "#b2182b", "#4daf4a"]  # blue, red, green
GT_SHADE = "#fee0d2"       # light red for GT scratch regions
GT_EDGE = "#de2d26"        # darker red edge
PR2_FILL = "#9ecae1"       # light blue for pr2 probability fill
PR2_LINE = "#2171b5"       # darker blue for pr2 line
THRESH_COLOR = "#636363"   # grey for 0.5 threshold

SF = 20  # sampling frequency (Hz)

# ---------- figure ----------
fig = plt.figure(figsize=(7.0, 3.0))  # single-column IMWUT width
gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 1], hspace=0.35, wspace=0.28,
                       left=0.07, right=0.98, top=0.92, bottom=0.10)

for col_idx, (title, seg_name) in enumerate(panels):
    seg_data = samples[samples.segment == seg_name].sort_values("timestep")
    t = seg_data["timestep"].values / SF  # convert to seconds

    # --- Top row: raw accelerometer signals ---
    ax_top = fig.add_subplot(gs[0, col_idx])

    # Shade ground-truth scratch regions
    gt = seg_data["gt_scratch"].values.astype(float)
    scratch_on = np.diff(np.concatenate(([0], gt, [0])))
    starts = np.where(scratch_on == 1)[0]
    ends = np.where(scratch_on == -1)[0]
    for s, e in zip(starts, ends):
        t_s = s / SF
        t_e = e / SF
        ax_top.axvspan(t_s, t_e, facecolor=GT_SHADE, edgecolor=GT_EDGE,
                       linewidth=0.5, alpha=0.7, zorder=0)

    # Plot x, y, z
    for ch_idx, ch in enumerate(["x", "y", "z"]):
        ax_top.plot(t, seg_data[ch].values, color=COLORS_XYZ[ch_idx],
                    linewidth=0.5, alpha=0.85, label=ch)

    ax_top.set_title(title, fontweight="bold", pad=4)
    if col_idx == 0:
        ax_top.set_ylabel("Accel. (g)")
        ax_top.legend(loc="upper right", ncol=3, frameon=False,
                      columnspacing=0.8, handlelength=1.2, borderpad=0)
    ax_top.set_xlim(t[0], t[-1])
    ax_top.tick_params(axis="x", labelbottom=False)

    # --- Bottom row: pr2 mask probabilities ---
    ax_bot = fig.add_subplot(gs[1, col_idx])

    pr2_prob = seg_data["pr2_prob"].values
    ax_bot.fill_between(t, 0, pr2_prob, color=PR2_FILL, alpha=0.7, zorder=1)
    ax_bot.plot(t, pr2_prob, color=PR2_LINE, linewidth=0.7, zorder=2)
    ax_bot.axhline(0.5, color=THRESH_COLOR, linestyle="--", linewidth=0.6,
                   zorder=3, label="Threshold")

    ax_bot.set_ylim(-0.02, 1.02)
    ax_bot.set_xlim(t[0], t[-1])
    ax_bot.set_xlabel("Time (s)")
    if col_idx == 0:
        ax_bot.set_ylabel("P(scratch)")

    # Show segment-level classification result
    row = summary[summary.segment == seg_name].iloc[0]
    pr1_text = f"P(seg)={row.pr1_prob:.2f}"
    ax_bot.text(0.97, 0.92, pr1_text, transform=ax_bot.transAxes,
                fontsize=6.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#999999", linewidth=0.5, alpha=0.85))

# Save
out_path = "/sessions/bold-dreamy-dijkstra/mnt/JNJ/papers/deep_scratch/fig/example_data.pdf"
fig.savefig(out_path, format="pdf", bbox_inches="tight")
print(f"\nSaved to {out_path}")

# Also save a PNG preview
png_path = "/sessions/bold-dreamy-dijkstra/fig8_preview.png"
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=200)
print(f"Preview at {png_path}")
plt.close()
