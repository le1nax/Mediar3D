import numpy as np
import tifffile
from pathlib import Path

def count_nonzero_per_slice(tiff_path):
    # Load the TIFF image
    img = tifffile.imread(tiff_path)  # shape: (Z, H, W) expected
    
    print(f"Loaded image shape: {img.shape}")
    
    # Check dimensions
    if img.ndim < 3:
        raise ValueError(f"Expected at least 3D (Z,H,W), got {img.shape}")
    
    # Count nonzero per slice
    counts = []
    for z in range(img.shape[0]):  # iterate 0th axis
        nonzero_count = np.count_nonzero(img[z])
        counts.append(nonzero_count)
        print(f"Slice {z}: {nonzero_count} nonzero pixels")
    
    return counts

if __name__ == "__main__":
    tiff_path = Path("../../Datasets/CTC/test_images/zc2dg/fs_inference_cellcenters/cell_0000_cellcenter.tiff")  # <-- change to your file
    counts = count_nonzero_per_slice(tiff_path)