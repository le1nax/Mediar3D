import tifffile as tiff
import numpy as np

def reslice_image(img, direction="z"):
    """
    Reslice 3D image so that the chosen axis becomes the new z-axis.
    img: numpy array (z, y, x)
    direction: "z", "y", or "x"
    """
    if direction == "z":      # keep as is
        return img
    elif direction == "y":    # make y the new z
        return np.transpose(img, (1, 0, 2))   # (z, y, x) -> (y, z, x)
    elif direction == "x":    # make x the new z
        return np.transpose(img, (2, 0, 1))   # (z, y, x) -> (x, z, y)
    else:
        raise ValueError("Direction must be 'x', 'y', or 'z'")

# --- Input ---
input_path = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test/t000.tiff"
output_path = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test_slice/slice550.tiff"
slice_index = 550
direction = "y"   # choose: "x", "y", or "z"

# --- Read ---
img = tiff.imread(input_path)
print(f"Original image shape: {img.shape}")  # usually (z, y, x)

# --- Reslice ---
resliced = reslice_image(img, direction=direction)
print(f"Resliced image shape ({direction}-axis as z): {resliced.shape}")

# --- Extract slice ---
if slice_index < resliced.shape[0]:
    slice_img = resliced[slice_index]
    tiff.imwrite(output_path, slice_img)
    print(f"Saved {direction}-resliced slice {slice_index} to {output_path}")
else:
    print(f"Slice index {slice_index} out of range (0–{resliced.shape[0]-1})")