import os
import pandas as pd
from pathlib import Path
import numpy as np

def summarize_ctc_results(base_dir, output_dir):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(base_dir.rglob("metrics*.csv"))
    if not csv_files:
        print("⚠️ No metrics CSV files found.")
        return

    all_results = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if not {"SEG", "IoU"}.issubset(df.columns):
            continue

        parts = csv_path.parts
        th_folder = [p for p in parts if p.startswith("th")]
        th_value = float(th_folder[0].replace("th", "")) if th_folder else None

        model_name = next((p for p in parts if "epoch" in p), "unknown_model")
        roi_folder = next((p for p in parts if p.startswith("ROI_")), "unknown_ROI")

        df["Model"] = model_name
        df["ROI"] = roi_folder
        df["Threshold"] = th_value

        all_results.append(df[["Model", "ROI", "Threshold", "SEG", "IoU"]])

    if not all_results:
        print("⚠️ No valid data found.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # --- Compute SEG boxplot stats using agg ---
    seg_stats = all_df.groupby(["Model", "ROI", "Threshold"]).agg(
        SEG_min=("SEG", "min"),
        SEG_Q1=("SEG", lambda x: np.percentile(x, 25)),
        SEG_median=("SEG", "median"),
        SEG_Q3=("SEG", lambda x: np.percentile(x, 75)),
        SEG_max=("SEG", "max")
    ).reset_index()

    # --- Compute mean IoU ---
    iou_mean = all_df.groupby(["Model", "ROI", "Threshold"])["IoU"].mean().reset_index().rename(columns={"IoU": "IoU_mean"})

    # Merge SEG stats + mean IoU
    summary_df = pd.merge(seg_stats, iou_mean, on=["Model", "ROI", "Threshold"], how="outer")

    summary_path = output_dir / "summary_seg_boxplot_iou_mean.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")

    print("\nColumns:")
    print(list(summary_df.columns))


# Example usage
if __name__ == "__main__":
    summarize_ctc_results(
        base_dir="/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_small_results/paper/01_final_test_small",
        output_dir="/work/scratch/geiger/Datasets/CTC/sim3d/01_final_test_small_results/paper/summary_boxplot"
    )


# import os
# import pandas as pd
# from pathlib import Path
# import re

# def summarize_ctc_results(base_dir, output_dir):
#     base_dir = Path(base_dir)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     for model_dir in base_dir.iterdir():
#         if not model_dir.is_dir():
#             continue

#         # Allow one or more underscores before ROI_
#         match = re.match(r"(.+?)_+ROI_\d+", model_dir.name)
#         if not match:
#             print(f"⚠️  Skipping {model_dir.name} (no ROI match)")
#             continue

#         arch_type = match.group(1)
#         arch_out_dir = output_dir / arch_type
#         arch_out_dir.mkdir(parents=True, exist_ok=True)

#         for roi_dir in model_dir.iterdir():
#             if not roi_dir.is_dir() or not roi_dir.name.startswith("ROI_"):
#                 continue

#             roi_name = roi_dir.name
#             summary_rows = []

#             for th_dir in roi_dir.iterdir():
#                 if not th_dir.is_dir() or not th_dir.name.startswith("th"):
#                     continue

#                 csv_files = list(th_dir.glob("metrics_*.csv"))
#                 if not csv_files:
#                     continue

#                 df = pd.read_csv(csv_files[0])
#                 if not {"SEG", "IoU"}.issubset(df.columns):
#                     print(f"Skipping {csv_files[0]} (missing SEG/IoU columns)")
#                     continue

#                 threshold = th_dir.name.replace("th", "")
#                 mean_seg = df["SEG"].mean()
#                 mean_iou = df["IoU"].mean()
#                 mean_ap05 = df["AP05"].mean() if "AP05" in df.columns else None

#                 row = {
#                     "threshold": threshold,
#                     "mean_SEG": mean_seg,
#                     "mean_IoU": mean_iou,
#                 }
#                 if mean_ap05 is not None:
#                     row["mean_AP05"] = mean_ap05
#                 summary_rows.append(row)

#             if summary_rows:
#                 out_df = pd.DataFrame(summary_rows)
#                 out_df.sort_values(by="threshold", inplace=True)
#                 out_path = arch_out_dir / f"{roi_name}.csv"
#                 out_df.to_csv(out_path, index=False)
#                 print(f"✅ Saved summary → {out_path}")

# # Example usage
# summarize_ctc_results(
#     base_dir="/work/scratch/geiger/Datasets/Blasto/final_test/final_results/condor/cropped_images",
#     output_dir="/work/scratch/geiger/Datasets/Blasto/final_test/final_results/condor/summary"
# )