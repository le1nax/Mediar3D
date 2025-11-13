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

def summarize_thresholds(base_dir, output_dir):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name.startswith("th")]
    print(f"🔍 Found {len(threshold_folders)} threshold folders.\n")

    all_results = []

    for folder in threshold_folders:
        threshold = get_threshold_from_folder(folder)
        csv_files = list(folder.glob("metrics*.csv"))
        if not csv_files:
            print(f"⚠️ No metrics CSV found in {folder.name}")
            continue

        # Concatenate all CSVs in this threshold folder
        dfs = [pd.read_csv(f)[["SEG"]] for f in csv_files]
        df_concat = pd.concat(dfs, ignore_index=True)

        # Compute SEG boxplot stats
        seg_stats = {
            "Threshold": threshold,
            "SEG": df_concat["SEG"].median(),
        }
        all_results.append(seg_stats)

    summary_df = pd.DataFrame(all_results).sort_values(by="Threshold").reset_index(drop=True)

    summary_path = output_dir / "summary_thresholds_seg_boxplot.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")
    print(summary_df)


if __name__ == "__main__":
    summarize_thresholds(
        base_dir="/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/condor/test/pretrained_mediar_00val_3gpus_5bs_2025-10-05_05-53-09_unwrapped",
        output_dir="/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/condor/test/pretrained_mediar_00val_3gpus_5bs_2025-10-05_05-53-09_unwrapped/summary_boxplot"
    )