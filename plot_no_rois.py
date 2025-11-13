import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation_thresholds(csv_path, figsize=(8, 5)):
    """
    Plots SEG (min, median, max) vs threshold for a single model (no ROI grouping),
    with a logarithmic x-axis.
    """
    # === Load Data ===
    df = pd.read_csv(csv_path)

    # --- Rename for clarity ---
    df.rename(columns={
        'SEG_min': 'Lower_SEG',
        'SEG_median': 'Median_SEG',
        'SEG_max': 'Highest_SEG'
    }, inplace=True)

    # --- Validate required columns ---
    required_cols = {'Threshold', 'Lower_SEG', 'Median_SEG', 'Highest_SEG'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected: {required_cols}")

    # --- Sort by threshold ---
    df = df.sort_values(by='Threshold')

    # === Plot ===
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['Threshold'], df['Median_SEG'], marker='o', color='tab:blue', label='Median SEG', linewidth=2)
    
    # Vertical bars for min/max SEG
    for x, y_low, y_high in zip(df['Threshold'], df['Lower_SEG'], df['Highest_SEG']):
        ax.plot([x, x], [y_low, y_high], color='tab:blue', alpha=0.4, linewidth=1)
        ax.plot(x, y_low, marker='_', color='tab:blue', markersize=8)
        ax.plot(x, y_high, marker='_', color='tab:blue', markersize=8)

    # === Formatting ===
    ax.set_xscale('log')  # <-- Logarithmic x-axis
    ax.set_xlabel('Cell Probability Threshold (log scale)', fontsize=12)
    ax.set_ylabel('Segmentation Score (SEG)', fontsize=12)
    ax.set_title('Segmentation Performance vs Threshold (Log Scale)', fontsize=14)
    ax.set_ylim(max(0.0, df['Lower_SEG'].min() - 0.05),
                min(1.0, df['Highest_SEG'].max() + 0.05))
    
    # Set nice major/minor gridlines
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

    print(f"✅ Chart generated (log-scale) for single-model SEG threshold performance across {len(df)} thresholds.")

# === Example usage ===
if __name__ == "__main__":
    csv_path = "/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res_SG/condor/summary_seg_boxplot_iou_mean.csv"
    plot_segmentation_thresholds(csv_path)