import os
import pandas as pd
from pathlib import Path
import numpy as np
import re

def get_threshold_from_folder(folder_path):
    folder_name = Path(folder_path).name
    match = re.search(r"th([0-9\.]+)", folder_name)
    if match:
        return float(match.group(1))
    return None

def round_down_to_magnitude(n):
    """Round down to nearest lower magnitude, e.g., 12 -> 10, 10003 -> 10000"""
    if n == 0:
        return 0
    magnitude = 10 ** (len(str(abs(n))) - 1)
    return int(n // magnitude * magnitude)

def summarize_ctc_results(base_dir, output_dir):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(base_dir.rglob("metrics*.csv"))
    print(f"🔍 Found {len(csv_files)} CSV files to process.\n")

    all_results = []

    for csv_path in csv_files:
        print(f"📂 Processing: {csv_path.parent}")
        df = pd.read_csv(csv_path)

        threshold = get_threshold_from_folder(csv_path.parent.parent)
        if threshold is None:
            print(f"⚠️ No threshold found in parent folder: {csv_path.parent.parent.name}")

        roi_match = re.search(r"ROI_(\d+)", csv_path.parent.name)
        roi_val = int(roi_match.group(1)) if roi_match else 0
        if "zero_shot" in csv_path.parent.name:
            roi_val = 0
        if roi_match is None and "zero_shot" not in csv_path.parent.name:
            print(f"⚠️ No ROI found in folder: {csv_path.parent.name}")

        roi_rounded = round_down_to_magnitude(roi_val)
        roi_rounded_string = f"ROI_{roi_rounded}"

        df["Model"] = Path(csv_path.parent.parent).name  # threshold folder
        df["ROI"] = roi_rounded_string
        df["Threshold"] = threshold

        all_results.append(df[["Model", "ROI", "Threshold", "SEG", "IoU"]])

    all_df = pd.concat(all_results, ignore_index=True)
    print(f"\n✅ Collected {len(all_df)} total rows from {len(csv_files)} files.")
    print(f"📊 Unique thresholds found: {sorted(all_df['Threshold'].dropna().unique())}")
    print(f"📊 Unique ROIs found: {sorted(all_df['ROI'].unique())}")

    # --- Compute SEG boxplot stats ---
    seg_stats = all_df.groupby(["ROI", "Threshold", "Model"]).agg(
        SEG_min=("SEG", "min"),
        SEG_Q1=("SEG", lambda x: np.percentile(x, 25)),
        SEG_median=("SEG", "median"),
        SEG_Q3=("SEG", lambda x: np.percentile(x, 75)),
        SEG_max=("SEG", "max")
    ).reset_index()

    # --- Compute mean IoU ---
    iou_mean = all_df.groupby(["ROI", "Threshold", "Model"])["IoU"].mean().reset_index().rename(columns={"IoU": "IoU_mean"})

    # Merge
    summary_df = pd.merge(seg_stats, iou_mean, on=["ROI", "Threshold", "Model"], how="outer")

    # --- Sort by ROI and Threshold ---
    summary_df = summary_df.sort_values(by=["ROI", "Threshold"]).reset_index(drop=True)

    summary_path = output_dir / "summary_seg_boxplot_iou_mean.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")
    print("Columns:", list(summary_df.columns))


if __name__ == "__main__":
    summarize_ctc_results(
        base_dir="/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/01_test_res_cpsam_all",
        output_dir="/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_res/01_test_res_cpsam_all/summary_boxplot"
    )