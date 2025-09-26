import os
import glob
import shutil

# Paths
base_dir = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/nuinsseg/archive"
output_images = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/nuinsseg_images"
output_masks = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/nuinsseg_masks"

# Create output dirs
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

# Collect all image + mask files
image_files = glob.glob(os.path.join(base_dir, "*", "tissue images", "*"))
mask_files = glob.glob(os.path.join(base_dir, "*", "label masks modify", "*"))

# Sort to ensure consistent pairing
image_files.sort()
mask_files.sort()

# Safety check
if len(image_files) != len(mask_files):
    print(f"⚠️ Warning: {len(image_files)} images but {len(mask_files)} masks found")

# Save with new names
for i, (img, mask) in enumerate(zip(image_files, mask_files)):
    new_img_name = f"cell_{i:05d}.tif"
    new_mask_name = f"cell_{i:05d}_label.tif"

    shutil.copy(img, os.path.join(output_images, new_img_name))
    shutil.copy(mask, os.path.join(output_masks, new_mask_name))

print(f"✅ Done! Saved {len(image_files)} images to {output_images} and {output_masks}")