import os
import numpy as np
import tifffile as tiff
from scipy.ndimage import label

# --- USER CONFIG ---
masks_npy = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/fold_2/Fold 2/masks/fold2/masks.npy"        # (N, H, W, 6)
images_npy = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/fold_2/Fold 2/images/fold2/images.npy"      # optional: for saving RGB images
out_masks = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/panNuke2_masks"
out_imgs  = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/panNuke2_img"  # optional: for saving RGB images

os.makedirs(out_masks, exist_ok=True)
os.makedirs(out_imgs, exist_ok=True)

# --- LOAD ---
X = np.load(images_npy) if os.path.exists(images_npy) else None
Y = np.load(masks_npy)  # shape (N, H, W, 6)

print("Masks shape:", Y.shape)
if X is not None:
    print("Images shape:", X.shape)

counter = 0

for i in range(Y.shape[0]):
    mask = Y[i]  # shape (H, W, 6)

    # Initialize empty instance mask (int32 to avoid overflow)
    instance_mask = np.zeros(mask.shape[:2], dtype=np.int32)
    next_id = 1

    # Loop over channels 0–4 (cell types)
    for ch in range(5):
        ch_mask = mask[..., ch]
        # label connected components
        labeled, num = label(ch_mask)
        if num > 0:
            labeled[labeled > 0] += next_id - 1  # shift IDs to keep unique
            instance_mask += labeled
            next_id = instance_mask.max() + 1

    # Skip empty patches (optional)
    if instance_mask.max() == 0:
        continue

    # Save instance mask as uint16
    mask_name = f"cell_{counter:05d}_label.tif"
    tiff.imwrite(os.path.join(out_masks, mask_name), instance_mask.astype(np.uint16))

    # Save corresponding image if available
    if X is not None:
        img_name = f"cell_{counter:05d}.tif"
        img = X[i].astype(np.uint8)
        tiff.imwrite(os.path.join(out_imgs, img_name), img)

    counter += 1

print(f"✅ Done! Saved {counter} instance-labeled masks.")
