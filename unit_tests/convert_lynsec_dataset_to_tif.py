#!/usr/bin/env python3
"""
Robust .npy -> .tif converter + diagnostic

Usage: edit the `input_folders` list and the output dirs below, then run.
"""

import os
import glob
import numpy as np
import tifffile as tiff
from pathlib import Path

# --- USER CONFIG ---
input_folders = [
    "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec/lynsec 1",
    "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec/lynsec 2",
    "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec/lynsec 3",
]

out_imgs = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec_img"
out_labels = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/lynsec_masks"

os.makedirs(out_imgs, exist_ok=True)
os.makedirs(out_labels, exist_ok=True)

# threshold for suspiciously small saved file (bytes)
SMALL_FILE_THRESHOLD = 1024  # 1 KB

# --- helper functions ---
def summarize_array(arr):
    try:
        s = f"shape={arr.shape}, dtype={arr.dtype}, min={np.nanmin(arr):g}, max={np.nanmax(arr):g}, mean={np.nanmean(arr):g}, nbytes={arr.nbytes}"
    except Exception:
        s = f"shape={getattr(arr,'shape',None)}, dtype={getattr(arr,'dtype',None)} (couldn't compute stats)"
    return s

def safe_load_npy(path):
    """Load npy; return ndarray. If memmap, convert to array to be safe for inspections."""
    arr = np.load(path, allow_pickle=False)  # disallow pickle for safety
    if isinstance(arr, np.memmap):
        arr = np.array(arr)
    return arr

def to_uint8_image(img):
    """
    Convert image to uint8 (H,W,3) or (H,W).
    - If float and max <= 1, assume 0..1 and scale by 255.
    - If float but max > 1 and <=255, clip and cast.
    - If >255, clip to 0..255.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # remove NaNs/Infs
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    if np.issubdtype(img.dtype, np.floating):
        mx = float(np.nanmax(img)) if img.size else 0.0
        if mx <= 1.0:
            img = (img * 255.0).round()
        # else: assume already in a larger dynamic range; we'll clip below
    # now clip and cast
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def to_uint16_mask(mask):
    """Convert mask to uint16 (round if float)."""
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    if np.issubdtype(mask.dtype, np.floating):
        mask = np.rint(mask)
    mask = np.clip(mask, 0, 65535).astype(np.uint16)
    return mask

def detect_channel_layout(arr):
    """
    Return "channels_first" or "channels_last" or None.
    Expected inputs: N-dimensional arrays:
      - (N, C, H, W) -> channels_first
      - (N, H, W, C) -> channels_last
    """
    if arr.ndim != 4:
        return None
    N, a, b, c = arr.shape
    # heuristic: channel dims are small (1-6) and H/W are larger
    for candidate in ("channels_first", "channels_last"):
        if candidate == "channels_first":
            C = a
            H = b
            W = c
        else:
            C = c
            H = a
            W = b
        if 1 <= C <= 6 and H > 10 and W > 10:
            return candidate
    return None

# --- main processing ---
counter = 0
for folder in input_folders:
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        print(f"⚠️ Input folder not found, skipping: {folder}")
        continue

    npy_files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not npy_files:
        print(f"⚠️ No .npy files in {folder}, skipping.")
        continue

    for npy_path in npy_files:
        print(f"\n--- processing file: {npy_path} ---")
        try:
            arr = safe_load_npy(npy_path)
        except Exception as e:
            print(f"  ❌ Failed to load {npy_path}: {e}")
            continue

        print("  file bytes:", os.path.getsize(npy_path))
        print("  array summary:", summarize_array(arr))

        # ensure arr is 4D: (N, C, H, W) or (N, H, W, C)
        if arr.ndim == 3:
            # could be single sample (C,H,W) or (H,W,C) or (N,H,W)
            # try to expand to (N, C, H, W) if possible
            # heuristics:
            s = arr.shape
            if s[0] in (1,3,4,5):  # likely (C,H,W)
                arr = arr[np.newaxis, ...]  # (1, C, H, W)
                print("  assumed original shape (C,H,W) -> expanded to (1,C,H,W)")
            elif s[-1] in (1,3,4,5):  # likely (H,W,C) -> (1,H,W,C)
                arr = arr[np.newaxis, ...]  # (1,H,W,C)
                print("  assumed original shape (H,W,C) -> expanded to (1,H,W,C)")
            else:
                # fallback: treat as (N,H,W)
                arr = arr[..., np.newaxis]  # (N,H,W,1)
                print("  assumed original shape (N,H,W) -> expanded to (N,H,W,1)")

        if arr.ndim != 4:
            print(f"  ❗ Unsupported ndarray dimension ({arr.ndim}). Skipping.")
            continue

        layout = detect_channel_layout(arr)
        if layout is None:
            # fallback: test both interpretations using small heuristics
            # prefer channels_first if arr.shape[1] in {3,4,5}
            if arr.shape[1] in (3,4,5):
                layout = "channels_first"
            elif arr.shape[-1] in (3,4,5):
                layout = "channels_last"
            else:
                layout = "channels_first"
            print(f"  heuristic layout chosen: {layout}")
        else:
            print(f"  detected channel layout: {layout}")

        N = arr.shape[0]
        for i in range(N):
            sample = arr[i]
            # get rgb and instance channels depending on layout
            if layout == "channels_first":
                # sample shape expected (C, H, W)
                if sample.shape[0] < 4:
                    print(f"   ⚠️ sample {i} has {sample.shape[0]} channels (<4). Skipping.")
                    continue
                rgb_raw = np.moveaxis(sample[0:3], 0, -1)  # -> (H,W,3)
                inst_raw = sample[3]                       # -> (H,W)
            else:
                # channels last: sample shape expected (H, W, C)
                if sample.shape[-1] < 4:
                    print(f"   ⚠️ sample {i} has {sample.shape[-1]} channels (<4). Skipping.")
                    continue
                rgb_raw = sample[..., 0:3]  # (H,W,3)
                inst_raw = sample[..., 3]   # (H,W)

            # safety: convert to numpy arrays
            rgb_raw = np.array(rgb_raw)
            inst_raw = np.array(inst_raw)

            # debug info per sample
            print(f"   sample {i} rgb: {summarize_array(rgb_raw)}")
            print(f"   sample {i} inst: {summarize_array(inst_raw)}")

            # convert/dtype handling
            rgb = to_uint8_image(rgb_raw)
            mask = to_uint16_mask(inst_raw)

            # filenames
            img_name = f"cell_{counter:05d}.tif"
            mask_name = f"cell_{counter:05d}_label.tif"
            img_path = os.path.join(out_imgs, img_name)
            mask_path = os.path.join(out_labels, mask_name)

            # save and then verify size
            try:
                tiff.imwrite(img_path, rgb)
                tiff.imwrite(mask_path, mask)
            except Exception as e:
                print(f"   ❌ Failed to save sample {counter}: {e}")
                continue

            # file size checks
            s_img = os.path.getsize(img_path)
            s_mask = os.path.getsize(mask_path)
            print(f"   saved {img_path} ({s_img} bytes), {mask_path} ({s_mask} bytes)")

            if s_img < SMALL_FILE_THRESHOLD or s_mask < SMALL_FILE_THRESHOLD:
                print("   ⚠️ Saved file suspiciously small — possible problem in saved arrays.")
                # print some extra verification output
                print("     -> rgb shape:", rgb.shape, "rgb.nbytes:", rgb.nbytes)
                print("     -> mask shape:", mask.shape, "mask.nbytes:", mask.nbytes)

            counter += 1

print(f"\n✅ Done. Exported {counter} images and masks.")