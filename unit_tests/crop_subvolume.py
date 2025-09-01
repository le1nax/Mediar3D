import tifffile as tiff
import os

# --- Input ---
input_path = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test/t000.tiff"
output_path = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test_subvolume/cropped.tiff"

# --- Ensure output directory exists ---
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# --- Read ---
img = tiff.imread(input_path)   # shape: (z, y, x)
print(f"Original image shape: {img.shape}")

# --- Crop ---
z_start, z_end = 300, 350
z_crop = img[z_start:z_end]

h, w = z_crop.shape[1], z_crop.shape[2]
cropped = z_crop[:, :h//2, :w//2]

print(f"Cropped image shape: {cropped.shape}")

# --- Save ---
tiff.imwrite(output_path, cropped)
print(f"Saved cropped volume to {output_path}")