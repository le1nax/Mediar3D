import os
import shutil
import tifffile as tiff
import numpy as np

# input + output dirs
input_dir = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_phasecontrast_img"
out_imgs = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_PhC_img"
out_labels = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/yeast_PhC_masks"

os.makedirs(out_imgs, exist_ok=True)
os.makedirs(out_labels, exist_ok=True)

# list all .tif files
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".tif")])

counter = 0
for f in all_files:
    if f.endswith("_mask.tif"):
        continue

    # find corresponding mask
    img_path = os.path.join(input_dir, f)
    mask_path = os.path.join(input_dir, f.replace("_im.tif", "_mask.tif"))

    if not os.path.exists(mask_path):
        print(f"⚠️ Warning: No mask found for {f}, skipping.")
        continue

    # load
    img = tiff.imread(img_path)
    mask = tiff.imread(mask_path)

    # check dimensionality
    if img.ndim == 2:  # 2D image
        new_img_name = f"cell_{counter:05d}.tif"
        new_mask_name = f"cell_{counter:05d}_label.tif"

        tiff.imwrite(os.path.join(out_imgs, new_img_name), img.astype(np.uint16))
        tiff.imwrite(os.path.join(out_labels, new_mask_name), mask.astype(np.uint16))

        counter += 1

    elif img.ndim == 3:  # 3D stack
        depth = img.shape[0]
        for z in range(depth):
            new_img_name = f"cell_{counter:05d}_z{z:03d}.tif"
            new_mask_name = f"cell_{counter:05d}_z{z:03d}_label.tif"

            tiff.imwrite(os.path.join(out_imgs, new_img_name), img[z].astype(np.uint16))
            tiff.imwrite(os.path.join(out_labels, new_mask_name), mask[z].astype(np.uint16))

        counter += 1

    else:
        print(f"⚠️ Unexpected shape for {f}: {img.shape}, skipping.")

print(f"✅ Done! Processed {counter} image/mask pairs.")