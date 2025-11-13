import os
import re
import glob
import csv
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from train_tools import *
from SetupDict import MODELS, PREDICTOR

from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast, average_precision_final
import tifffile as tif


# def get_res_ctc(out_dir, gt_path):
#     csv_path = os.path.join(out_dir, "metrics.csv")

#     # 🔥 Remove any existing CSV files in the folder
#     for old_csv in glob.glob(os.path.join(out_dir, "*.csv")):
#         try:
#             os.remove(old_csv)
#             print(f"🗑️ Removed old CSV: {old_csv}")
#         except Exception as e:
#             print(f"⚠️ Could not remove {old_csv}: {e}")

#     # Create new CSV file with header
#     header = ["image_name", "t_idx", "z_idx", "SEG", "IoU", "AP05"]
#     with open(csv_path, "w", newline="") as f:
#         csv.writer(f).writerow(header)

#     # Get all prediction and GT files
#     pred_files = {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(out_dir, "t*_label.tif*")))}
#     gt_files = sorted(glob.glob(os.path.join(gt_path, "man_seg_*.tif*")))

#     for gt_file in tqdm(gt_files, desc="Evaluating"):
#         gt_name = os.path.basename(gt_file)
#         match = re.match(r"man_seg_(\d{3})_(\d{3})\.tif", gt_name)
#         if not match:
#             print(f"⚠️ Skipping {gt_name}: invalid name pattern")
#             continue

#         t_str, z_str = match.groups()
#         t_idx, z_idx = int(t_str), int(z_str)

#         # Corresponding prediction volume
#         pred_name = f"t{t_str}_label.tiff"
#         if pred_name not in pred_files:
#             print(f"⚠️ Missing prediction for time {t_str}")
#             continue

#         # Load volumes/slices
#         pred_vol = tif.imread(pred_files[pred_name])
#         if z_idx >= pred_vol.shape[0]:
#             print(f"⚠️ z={z_idx} out of range for {pred_name}")
#             continue

#         pred_slice = pred_vol[z_idx]
#         gt_mask = tif.imread(gt_file)

#         # Compute metrics
#         seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_slice)
#         AP05_score = average_precision_final(gt_mask, pred_slice)
#         iou, _, _, _ = evaluate_metrics_cellseg(gt_mask, pred_slice)

#         # Write results
#         with open(csv_path, "a", newline="") as f:
#             csv.writer(f).writerow([gt_name, t_idx, z_idx, seg_score, iou, AP05_score])

#     print(f"✅ Metrics saved: {csv_path}")


# Prepare CSV path for this threshold
def get_res(out_dir, gt_path):
    # --- Delete any existing CSVs in the folder ---
    old_csvs = glob.glob(os.path.join(out_dir, "*.csv"))
    for csv_file in old_csvs:
        try:
            os.remove(csv_file)
            print(f"🧹 Removed old CSV: {csv_file}")
        except Exception as e:
            print(f"⚠️ Could not delete {csv_file}: {e}")

    # --- Create a new metrics.csv ---
    csv_path = os.path.join(out_dir, "metrics.csv")
    header = ["image_name", "SEG", "IoU", "AP05"]

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    # --- Evaluate predictions ---
    pred_files = sorted(glob.glob(os.path.join(out_dir, "*.tif*")))

    for pred_path in tqdm(pred_files, desc=f"Evaluating {os.path.basename(out_dir)}"):
        image_name = os.path.basename(pred_path)
        gt_file = os.path.join(gt_path, image_name)

        if not os.path.exists(gt_file):
            print(f"⚠️ Skipping {image_name} (GT missing)")
            continue

        pred_mask = tif.imread(pred_path)
        gt_mask = tif.imread(gt_file)

        seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_mask)
        AP05_score = average_precision_final(gt_mask, pred_mask)
        iou, _, _1, _2 = evaluate_metrics_cellseg(gt_mask, pred_mask)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([image_name, seg_score, iou, AP05_score])

    print(f"✅ Metrics saved: {csv_path}")

# if __name__ == "__main__":
#     res_dir = "/work/scratch/geiger/Datasets/Blasto/final_test/final_results/cpsam_thresholds"
#     gt_dir = "/work/scratch/geiger/Datasets/Blasto/final_test/cropped_masks"

#     for main_folder in os.listdir(res_dir):
#         main_path = os.path.join(res_dir, main_folder)
#         if not os.path.isdir(main_path):
#             continue

#         # Iterate over subfolders inside each main folder
#         for sub_folder in os.listdir(main_path):
#             sub_path = os.path.join(main_path, sub_folder)
#             if os.path.isdir(sub_path):
#                 print(f"📂 Evaluating: {sub_path}")
#                 get_res(sub_path, gt_dir)


if __name__ == "__main__":
    res_dir = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/final_results/cpsam"
    for folder in os.listdir(res_dir):
        folder_path = os.path.join(res_dir, folder)
        if os.path.isdir(folder_path):  # only iterate over directories
            get_res(folder_path, "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/GT")

# if __name__ == "__main__":
#     base_dir = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-DRO/02_final_test_res/paper_morethresholds/02_final_test"
#     gt_dir = "/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2020/Fluo-N3DL-DRO/02_GT/SEG"
#     ctc = True

#     for model_folder in os.listdir(base_dir):
#         model_path = os.path.join(base_dir, model_folder)
#         if not os.path.isdir(model_path):
#             continue

#         # Each model folder should contain exactly one ROI_xxx folder
#         roi_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
#         for roi_folder in roi_folders:
#             roi_path = os.path.join(model_path, roi_folder)

#             # Now inside the ROI folder, iterate over threshold subfolders (leaves)
#             threshold_folders = [f for f in os.listdir(roi_path) if os.path.isdir(os.path.join(roi_path, f))]
#             for th_folder in threshold_folders:
#                 th_path = os.path.join(roi_path, th_folder)
#                 print(f"Processing: {th_path}")
#                 if(ctc):
#                     get_res_ctc(th_path, gt_dir)
#                 else:
#                     get_res(th_path, gt_dir)

# import os
# import re
# import glob
# import csv
# import torch
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime

# from train_tools import *
# from SetupDict import MODELS, PREDICTOR
# from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast, average_precision_final
# import tifffile as tif


# # Prepare CSV path for this threshold
# def get_res(out_dir, gt_path):
#     # ✅ Remove any existing CSV file in this leaf folder
#     existing_csvs = glob.glob(os.path.join(out_dir, "*.csv"))
#     for csv_file in existing_csvs:
#         try:
#             os.remove(csv_file)
#             print(f"🧹 Removed old CSV: {csv_file}")
#         except Exception as e:
#             print(f"⚠️ Could not remove {csv_file}: {e}")

#     # Create new CSV named metrics.csv
#     csv_path = os.path.join(out_dir, "metrics.csv")
#     header = ["image_name", "SEG", "IoU", "AP05"]

#     with open(csv_path, "w", newline="") as f:
#         csv.writer(f).writerow(header)

#     # Evaluate predictions
#     pred_files = sorted(glob.glob(os.path.join(out_dir, "*.tif*")))

#     for pred_path in tqdm(pred_files, desc=f"Evaluating {os.path.basename(out_dir)}"):
#         image_name = os.path.basename(pred_path)
#         gt_file = os.path.join(gt_path, image_name)

#         if not os.path.exists(gt_file):
#             print(f"⚠️ Skipping {image_name} (GT missing)")
#             continue

#         pred_mask = tif.imread(pred_path)
#         gt_mask = tif.imread(gt_file)

#         seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_mask)
#         AP05_score = average_precision_final(gt_mask, pred_mask)
#         iou, _, _1, _2 = evaluate_metrics_cellseg(gt_mask, pred_mask)

#         with open(csv_path, "a", newline="") as f:
#             csv.writer(f).writerow([image_name, seg_score, iou, AP05_score])

#     print(f"✅ Metrics saved: {csv_path}")


# if __name__ == "__main__":
#     base_dir = "/work/scratch/geiger/Datasets/Blasto/final_test/final_results/cpsam_thresholds/cp_sam_th0.9"
#     gt_dir = "/work/scratch/geiger/Datasets/Blasto/final_test/cropped_masks"

#     for model_folder in os.listdir(base_dir):
#         model_path = os.path.join(base_dir, model_folder)
#         if not os.path.isdir(model_path):
#             continue

#         # Each model folder should contain exactly one ROI_xxx folder
#         roi_folders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
#         for roi_folder in roi_folders:
#             roi_path = os.path.join(model_path, roi_folder)

#             # Now inside the ROI folder, iterate over threshold subfolders (leaves)
#             threshold_folders = [f for f in os.listdir(roi_path) if os.path.isdir(os.path.join(roi_path, f))]
#             for th_folder in threshold_folders:
#                 th_path = os.path.join(roi_path, th_folder)
#                 print(f"\n📂 Processing: {th_path}")
#                 get_res(th_path, gt_dir)