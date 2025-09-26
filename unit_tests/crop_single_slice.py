import tifffile as tiff
import os

# Input: can be a single .tif file or a directory of .tif files
input_path = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/test/cell_0001.tiff"
input_path_mask = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/GT/cell_0001_label.tiff"

output_dir = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/temp_3dimg"
output_dir_mask = "/work/scratch/geiger/Datasets/CTC/test_images/zc2dg/temp_3dmasks"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

# Define output file paths
output_file = os.path.join(output_dir, "cell_00002.tiff")
output_file_mask = os.path.join(output_dir_mask, "cell_00002_label.tiff")

# Read data
img = tiff.imread(input_path)
mask = tiff.imread(input_path_mask)

# Take last slice
img = img[:,:,100]
mask = mask[:,:,100]

# Save files
tiff.imwrite(output_file, img)
tiff.imwrite(output_file_mask, mask)