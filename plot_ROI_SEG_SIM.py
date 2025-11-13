import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Define CSV paths and thresholds for each model ---
models = {
    "Mediar + Paper Weights": {
        "path": "/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/paper/summary_boxplots/summary_seg_boxplot_iou_mean_paperweights.csv",
        "threshold": 0.01,
    },
    "Mediar + SAM2 Encoder": {
        "path":  "/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/Hiera/summary_boxplots/summary_seg_boxplot_iou_mean_hiera.csv",
        "threshold": 0.3,
    },
    "Mediar + Extended Pretraining": {
        "path": "/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/condor_high_th/summary_boxplots/summary_seg_boxplot_iou_mean_higher_thresholds.csv",
        "threshold": 0.5,
    },
    "CellposeSAM": {
        "path": "/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/01_test_res_cpsam_all/summary_boxplot/summary_seg_boxplot_iou_mean_cellpose.csv",
        "threshold": 0.0,
    },
}

# --- Visualization setup ---
plt.rcParams['font.family'] = 'serif'
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 20
BAR_LINEWIDTH = 2.0
MEDIAN_LINEWIDTH = 3.5  # ⬅️ Thicker median line
QMARKER_SIZE = 6

# --- Color palette ---
colors = plt.cm.Dark2(np.linspace(0, 1, len(models)))

# --- Define log offset per model (to separate overlapping bars) ---
LOG_OFFSET = 0.05
MODEL_OFFSETS = {}
for i, model in enumerate(models.keys()):
    MODEL_OFFSETS[model] = (i - len(models) / 2) * LOG_OFFSET

# --- Helper function for log-space shifting ---
def get_shifted_x(x, log_offset):
    """Applies a small additive offset in log space and returns shifted value."""
    if x <= 0:
        x = 0.5  # avoid log(0)
    return 10 ** (np.log10(x) + log_offset)

# --- Create plot ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_title('SEG vs. Training ROIs (Model-specific thresholds)', fontsize=TITLE_FONTSIZE, pad=10)

all_rois = set()

# --- Iterate through each model ---
for (model_name, model_info), color in zip(models.items(), colors):
    file_path = model_info["path"]
    threshold_value = model_info["threshold"]
    log_offset = MODEL_OFFSETS[model_name]

    # Load CSV
    df = pd.read_csv(file_path)

    # Normalize column names
    rename_map = {
        'SEG_min': 'Lower_SEG',
        'SEG_median': 'Median_SEG',
        'SEG_max': 'Highest_SEG',
        'SEG_q1': 'SEG_Q1',
        'SEG_Q1': 'SEG_Q1',
        'SEG_q3': 'SEG_Q3',
        'SEG_Q3': 'SEG_Q3'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    df['ROI_value'] = df['ROI'].str.extract(r'(\d+)').astype(float)
    all_rois.update(df['ROI_value'].unique())
    df['ROI_plot'] = df['ROI_value'].replace(0, 0.5)
    df['ROI_shifted'] = df['ROI_plot'].apply(lambda x: get_shifted_x(x, log_offset))

    df = df[df['Threshold'] == threshold_value].copy()
    if df.empty:
        print(f"⚠️ No entries found for {model_name} at Threshold = {threshold_value}")
        continue

    seg_cols = ['Lower_SEG', 'SEG_Q1', 'Median_SEG', 'SEG_Q3', 'Highest_SEG']
    for c in seg_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.sort_values(by='ROI_shifted')

    # --- Plot faint quartile and min/max info ---
    for _, row in df.iterrows():
        x = row['ROI_shifted']

        # Min–max line (very faint)
        ax.plot([x, x], [row['Lower_SEG'], row['Highest_SEG']],
                color=color, alpha=0.15, linewidth=BAR_LINEWIDTH)

        # Q1–Q3 range (slightly stronger)
        ax.plot([x, x], [row['SEG_Q1'], row['SEG_Q3']],
                color=color, alpha=0.25, linewidth=BAR_LINEWIDTH + 0.5)

        # Quartile + min/max markers (faint)
        ax.scatter(x, row['Lower_SEG'], color=color, marker='_', s=QMARKER_SIZE * 4, alpha=0.25)
        ax.scatter(x, row['SEG_Q1'], color=color, marker='_', s=QMARKER_SIZE * 4, alpha=0.25)
        ax.scatter(x, row['SEG_Q3'], color=color, marker='_', s=QMARKER_SIZE * 4, alpha=0.25)
        ax.scatter(x, row['Highest_SEG'], color=color, marker='_', s=QMARKER_SIZE * 4, alpha=0.25)

    # --- Strong median line ---
    ax.plot(df['ROI_shifted'], df['Median_SEG'],
            color=color, linewidth=MEDIAN_LINEWIDTH, marker='o',
            label=f"{model_name} (th={threshold_value:g})", alpha=0.95)

# --- Axis config ---
ax.set_xlabel('#ROIs seen during training', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel('SEG-Score', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_xscale('log')

def roi_log_formatter(x, pos):
    if np.isclose(x, 0.5):
        return '0'
    elif x >= 1000:
        return f'{int(x):,}'.replace(',', ' ')
    else:
        return f'{int(x)}'

ax.xaxis.set_major_formatter(ticker.FuncFormatter(roi_log_formatter))

xticks = sorted(all_rois)
xticks = [0.5 if v == 0 else v for v in xticks]
ax.set_xticks(xticks)

# --- Y-limits ---
all_vals = []
for line in ax.lines:
    y = line.get_ydata()
    if len(y) > 0:
        all_vals.extend(y)
if all_vals:
    ax.set_ylim(max(0.0, min(all_vals) - 0.05), min(1.0, max(all_vals) + 0.05))

# --- Grid & legend ---
ax.grid(True, linestyle='--', alpha=0.5, which='major')
ax.legend(loc='lower right', fontsize=12, title='Model (threshold)', title_fontsize=14)

plt.tight_layout()
plt.savefig('/home/students/geiger/Downloads/plots/seg_vs_roi_multi_model_log_offset_clean.png', dpi=300)
plt.show()