import os
import numpy as np
import tifffile as tiff

# --- INPUT FILES ---
images_npy = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/images.npy"
masks_npy = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/labels.npy"

# --- OUTPUT DIRS ---
out_imgs = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/CoNIC_img"
out_masks = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/CoNIC_masks"
os.makedirs(out_imgs, exist_ok=True)
os.makedirs(out_masks, exist_ok=True)

# --- LOAD ---
X = np.load(images_npy)  # shape (N, H, W, 3)
Y = np.load(masks_npy)   # shape (N, H, W, 2)

print("Loaded:")
print("  Images:", X.shape)
print("  Masks: ", Y.shape)

# --- PROCESS ---
counter = 0
skipped = 0

for i in range(X.shape[0]):
    rgb = X[i]              # (H, W, 3)
    inst = Y[i, :, :, 0]    # (H, W) instance mask only

    # skip empty instance masks
    if np.all(inst == 0):
        skipped += 1
        continue

    img_name  = f"cell_{counter:05d}.tif"
    mask_name = f"cell_{counter:05d}_label.tif"

    # save rgb (convert to uint8 just in case)
    tiff.imwrite(os.path.join(out_imgs, img_name), rgb.astype(np.uint8))

    # save instance mask as HW image
    tiff.imwrite(os.path.join(out_masks, mask_name), inst.astype(np.uint16))

    counter += 1

print(f"\n✅ Done! Saved {counter} image/mask pairs, skipped {skipped} empty patches.")