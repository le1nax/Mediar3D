import os
import re
import glob
import csv
import torch
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime

from train_tools import *
from SetupDict import MODELS, PREDICTOR


from train_tools.measures import evaluate_metrics_cellseg, compute_CTC_SEG_fast, average_precision_final
import tifffile as tif

out_dir = "/work/scratch/geiger/Datasets/Blasto/final_test/results_final/paper/cropped_images/MediarFT_paper_blasto__ROI_100000_epoch23/ROI_100000/th0.0001"
gt_path = "/work/scratch/geiger/Datasets/Blasto/final_test/cropped_masks/"
csv_path =     "/work/scratch/geiger/Datasets/Blasto/final_test/results_final/paper/cropped_images/MediarFT_paper_blasto__ROI_100000_epoch23/ROI_100000/th0.0001/metrics_MediarFT_paper_blasto__ROI_100000_epoch23_ROI100000_th0.0001.csv"

#Evaluate predictions
pred_files = sorted(glob.glob(os.path.join(out_dir, "*.tif*")))

for pred_path in tqdm(pred_files, desc=f"Evaluating "):
    image_name = os.path.basename(pred_path)
    gt_file = os.path.join(gt_path, image_name)
    if not os.path.exists(gt_file):
        print(f"⚠️ Skipping {image_name} (GT missing)")
        continue

    pred_mask = tif.imread(pred_path)
    gt_mask = tif.imread(gt_file)

    seg_score, _ = compute_CTC_SEG_fast(gt_mask, pred_mask)
    AP05_score = average_precision_final(gt_mask, pred_mask)
    iou,_,_1,_2 = evaluate_metrics_cellseg(gt_mask, pred_mask)

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([image_name, seg_score, iou, AP05_score])