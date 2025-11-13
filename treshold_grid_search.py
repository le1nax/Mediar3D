import os
import glob
import csv
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import pandas as pd
import tifffile as tif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from train_tools import *
from SetupDict import MODELS, PREDICTOR
from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast

def run_single_model_prediction(config_path):
    # =====================================================
    # 1. Load Base Config
    # =====================================================
    opt = ConfLoader(config_path).opt
    setups = opt.pred_setups
    pprint_config(opt)

    model_path = setups.model_path        # Now this points directly to ONE .pth file
    thresholds = setups.algo_params["cellprob_thresholds"]
    input_path = setups.input_path
    gt_path = setups.ground_truth_path
    device = setups.device
    output_root = setups.output_path

    dataset_name = os.path.basename(os.path.normpath(input_path))
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"🔍 Evaluating single model: {model_name}")

    # =====================================================
    # 2. Load Model
    # =====================================================
    model_args = setups.model
    model = MODELS[model_args.name](**model_args.params)
    weights = torch.load(model_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    # =====================================================
    # 3. Predictor Object
    # =====================================================
    predictor = PREDICTOR[setups.name](
        model,
        device,
        input_path,
        None,  # output_path set dynamically below
        "",
        setups.make_submission,
        setups.exp_name,
        setups.algo_params,
    )

    # =====================================================
    # 4. Run Threshold Grid Search
    # =====================================================
    for th in thresholds:
        print(f"\n→ Threshold: {th}")
        out_dir = os.path.join(output_root, dataset_name, model_name, f"th{th}")
        os.makedirs(out_dir, exist_ok=True)
        predictor.output_path = out_dir
        predictor.cellprob_threshold = th

        # --- Run prediction ---
        predictor.conduct_prediction()

        # --- Prepare CSV ---
        csv_path = os.path.join(out_dir, f"metrics_{model_name}_th{th}.csv")
        header = ["image_name", "SEG", "IoU"]
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

        # --- Evaluate ---
        pred_files = sorted(glob.glob(os.path.join(out_dir, "*.tif*")))
        for pred_path in tqdm(pred_files, desc=f"Evaluating {model_name} @ {th}"):
            image_name = os.path.basename(pred_path)
            gt_file = os.path.join(gt_path, image_name)
            if not os.path.exists(gt_file):
                print(f"Skipping {image_name} (GT missing)")
                continue

            pred_mask = tif.imread(pred_path)
            gt_mask = tif.imread(gt_file)

            seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_mask)
            iou, _, _, _ = evaluate_metrics_cellseg(gt_mask, pred_mask)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([image_name, seg_score, iou])

        print(f"✅ Metrics saved: {csv_path}")

    print("\nAll thresholds completed.")

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



def summarize_ctc_results(base_dir, output_dir):
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(base_dir.rglob("metrics*.csv"))
    if not csv_files:
        print("No metrics CSV files found.")
        return

    all_results = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if not {"SEG", "IoU"}.issubset(df.columns):
            continue

        th_folder = [p for p in csv_path.parts if p.startswith("th")]
        th_value = float(th_folder[0].replace("th", "")) if th_folder else None
        model_name = next((p for p in csv_path.parts if not p.startswith("th") and p.endswith(".pth") == False), "unknown_model")

        df["Model"] = model_name
        df["Threshold"] = th_value
        all_results.append(df[["Model", "Threshold", "SEG", "IoU"]])

    if not all_results:
        print("⚠️ No valid data found.")
        return

    all_df = pd.concat(all_results, ignore_index=True)

    # --- Compute SEG stats + mean IoU ---
    seg_stats = all_df.groupby(["Model", "Threshold"]).agg(
        SEG_min=("SEG", "min"),
        SEG_Q1=("SEG", lambda x: np.percentile(x, 25)),
        SEG_median=("SEG", "median"),
        SEG_Q3=("SEG", lambda x: np.percentile(x, 75)),
        SEG_max=("SEG", "max")
    ).reset_index()

    iou_mean = all_df.groupby(["Model", "Threshold"])["IoU"].mean().reset_index().rename(columns={"IoU": "IoU_mean"})
    summary_df = pd.merge(seg_stats, iou_mean, on=["Model", "Threshold"], how="outer")

    summary_path = output_dir / "summary_seg_boxplot_iou_mean.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary: {summary_path}")
    return summary_path


if __name__ == "__main__":
    CONFIG_PATH = "./config/step3_prediction/grid_predictions.json"
    run_single_model_prediction(CONFIG_PATH)

    # Summarize results
    opt = ConfLoader(CONFIG_PATH).opt
    setups = opt.pred_setups
    output_root = setups.output_path
    summary_path = summarize_ctc_results(output_root, output_root)
    plot_segmentation_thresholds(summary_path)