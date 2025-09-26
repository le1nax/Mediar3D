import tifffile as tiff
import SimpleITK as sitk
import os

# Input TIFF
input_path = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/GT/cell_0001_label.tiff"

# Output MHD file
output_path = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/GT_MHD/cell_0001_label.mhd"
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # <-- create parent folder

# Load TIFF
img = tiff.imread(input_path)

# Convert to SimpleITK image
sitk_img = sitk.GetImageFromArray(img)

# Write MHD + RAW
sitk.WriteImage(sitk_img, output_path)

print(f"Saved {output_path} (with accompanying .raw file)")