import os
import numpy as np
from skimage import morphology, measure
import tifffile as tiff
from glob import glob

def process_mask(mask: np.ndarray, min_size: int = 16, hole_conn: int = 1):
    """
    Apply morphological post-processing to a 3D mask.
    
    Args:
        mask (np.ndarray): Input mask (3D array).
        min_size (int): Minimum object size (in voxels) to keep.
        hole_conn (int): Connectivity for small hole removal.
    
    Returns:
        np.ndarray: Labeled mask (3D array of ints).
    """
    # Binarize
    mask = mask > 0.5

    # Remove small holes (3D-aware)
    mask = morphology.remove_small_holes(mask, connectivity=hole_conn)

    # Remove small objects (3D-aware)
    mask = morphology.remove_small_objects(mask, min_size)

    # Relabel connected components
    mask = measure.label(mask)

    return mask

if __name__ == "__main__":
    # Input: can be a single .tif file or a directory of .tif files
    input_path = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-DRO/02_test_res_cpsam"
    output_dir = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-DRO/02_test_res_cpsam_postprocessing"
    os.makedirs(output_dir, exist_ok=True)

    # Collect input files
    if os.path.isdir(input_path):
        files = sorted(glob(os.path.join(input_path, "*.tif*")))
    else:
        files = [input_path]

    print(f"Found {len(files)} file(s)")

    for f in files:
        print(f"Processing {f}")
        mask = tiff.imread(f)  # shape (Z, Y, X)
        print("Loaded mask shape:", mask.shape)

        processed = process_mask(mask)

        # Save with _processed suffix
        filename = os.path.basename(f)
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".gz":  # handle .tiff.gz case
            name, ext2 = os.path.splitext(name)
            ext = ext2 + ext
        out_path = os.path.join(output_dir, f"{name}_processed{ext}")

        tiff.imwrite(out_path, processed.astype(np.uint16))
        print(f"Saved processed mask to {out_path}")