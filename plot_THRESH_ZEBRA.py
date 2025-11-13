import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Define your CSV files and labels ---
csv_files = {
    'Mediar + Paper Weights': '/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/paper/test/from_phase1/summary_boxplot/summary_thresholds_seg_boxplot.csv',
    'Mediar + Extended Pretraining': '/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/condor/test/pretrained_mediar_00val_3gpus_5bs_2025-10-05_05-53-09_unwrapped/summary_boxplot/summary_thresholds_seg_boxplot.csv',
    'Mediar + SAM2 Encoder': '/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/hiera/test/pretrained_hiera_0val_4gpus_5bs_2025-10-02_15-57-51_unwrapped/summary_boxplot/summary_thresholds_seg_boxplot.csv',
    'CellposeSAM': '/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/cpsam/summary_boxplot/summary_thresholds_seg_boxplot.csv'
}

# --- Visualization Parameters ---
plt.rcParams['font.family'] = 'serif'
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 20
LINE_WIDTH = 2.5

# --- Color palette for 4 curves ---
colors = plt.cm.Dark2(np.linspace(0, 1, len(csv_files)))

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.set_title('SEG vs Threshold', fontsize=TITLE_FONTSIZE, pad=10)

# --- Plot each CSV ---
for (label, path), color in zip(csv_files.items(), colors):
    try:
        df = pd.read_csv(path)
        df = df.sort_values(by="Threshold").reset_index(drop=True)

        # Plot median SEG
        ax.plot(
            df["Threshold"],
            df["SEG"],
            color=color,
            marker='o',
            linestyle='-',
            linewidth=LINE_WIDTH,
            label=label
        )

        # Add min/max vertical bars
        for x, y_min, y_max in zip(df["Threshold"], df["SEG"], df["SEG"]):
            ax.plot([x, x], [y_min, y_max], color=color, alpha=0.3, linewidth=1.2)

    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
    except Exception as e:
        print(f"⚠️ Could not process {path}: {e}")

# --- Axis Configuration ---
ax.set_xlabel('Cell Probability Threshold', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel('SEG-Score', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_xscale('log')

# Define custom formatter for log axis
def custom_log_formatter(x, pos):
    if x < 0.1 and x > 0 and (np.log10(x) % 1 == 0 or np.isclose(np.log10(x) % 1, 0)):
        return f'${{10^{{{np.log10(x):.0f}}}}}$'
    return f'{x:g}'

ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(custom_log_formatter))
ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
ax.grid(True, linestyle='--', alpha=0.6)

# --- Y limits ---
min_y = min(df["SEG"].min() for df in [pd.read_csv(f) for f in csv_files.values()])
max_y = max(df["SEG"].max() for df in [pd.read_csv(f) for f in csv_files.values()])
ax.set_ylim(max(0.0, min_y - 0.05), min(1.0, max_y + 0.05))

# --- Legend ---
ax.legend(loc='lower right', fontsize=12)

# --- Save ---
plt.tight_layout()
plt.savefig('/home/students/geiger/Downloads/plots/seg_vs_threshold_all_models.png', dpi=300)
plt.show()
