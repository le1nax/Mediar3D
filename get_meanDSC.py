import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
from tqdm import tqdm

def fast_mean_dice(pred, gt):
    """Compute mean Dice only on annotated GT cells using vectorized overlap."""
    # Flatten arrays
    pred_flat = pred.ravel()
    gt_flat = gt.ravel()

    # Only consider non-background GT
    gt_labels = np.unique(gt_flat)
    gt_labels = gt_labels[gt_labels != 0]

    if len(gt_labels) == 0:
        return np.nan

    # Compute contingency table
    max_pred = pred_flat.max() + 1
    max_gt = gt_flat.max() + 1
    table = np.zeros((max_gt, max_pred), dtype=np.int64)
    np.add.at(table, (gt_flat, pred_flat), 1)

    dice_scores = []
    for label_id in gt_labels:
        # Exclude background in pred
        overlap = table[label_id, 1:]
        if overlap.sum() == 0:
            continue
        pred_label = overlap.argmax() + 1  # +1 because we excluded background
        intersection = table[label_id, pred_label]
        size_sum = table[label_id, :].sum() + table[:, pred_label].sum()
        dice_scores.append(2 * intersection / size_sum)

    if len(dice_scores) == 0:
        return np.nan
    return np.mean(dice_scores)

# ---- Configure paths ----
pred_dir = Path("/work/scratch/geiger/Datasets/CTC/test_images/PlantSeg/final_results/hiera/images/pretrained_hiera_0val_4gpus_5bs_2025-10-02_15-57-51_unwrapped/th0.001")
gt_dir = Path("/netshares/BiomedicalImageAnalysis/Resources/Arabidopsis3DDigitalTissueAtlas_WolnyBioRxiv/segmentation")
out_csv = pred_dir / "dice_results_fast.csv"

# ---- Run evaluation ----
results = []
for pred_file in tqdm(sorted(pred_dir.glob("*.tiff"))):
    base_name = pred_file.stem.replace("_label", "")
    gt_candidates = list(gt_dir.glob(f"{base_name}*"))
    if len(gt_candidates) == 0:
        print(f"⚠️ No GT found for {pred_file.name}")
        continue
    gt_file = gt_candidates[0]

    try:
        pred = tifffile.imread(pred_file)
        gt = tifffile.imread(gt_file)
    except Exception as e:
        print(f"❌ Error reading {pred_file.name} or {gt_file.name}: {e}")
        continue

    if pred.shape != gt.shape:
        print(f"⚠️ Shape mismatch: {pred_file.name} vs {gt_file.name}")
        continue

    dsc = fast_mean_dice(pred, gt)
    results.append({"filename": pred_file.name, "mean_dice": dsc})

# ---- Save results ----
if results:
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved results to: {out_csv.resolve()}")
    valid_dsc = df["mean_dice"].dropna()
    if len(valid_dsc) > 0:
        print(f"Average Dice: {valid_dsc.mean():.4f} over {len(valid_dsc)} images")
else:
    print("❌ No valid Dice scores computed.")