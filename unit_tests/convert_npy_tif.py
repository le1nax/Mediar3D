import os
import numpy as np
import tifffile as tiff

# --- USER: set your folder path ---
folder = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/BlastoSPIM2_test_lowSNR/Blast_035/masks"

for fname in os.listdir(folder):
    if fname.endswith(".npy"):
        npy_path = os.path.join(folder, fname)
        arr = np.load(npy_path)

        # squeeze singleton dims (if e.g. (1, H, W))
        arr = np.squeeze(arr)

        # choose dtype for saving
        if np.issubdtype(arr.dtype, np.floating):
            # assume floats are scaled 0..1, rescale to uint8
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            # leave integer types as-is, cast to uint16 for safety
            arr = arr.astype(np.uint16)

        out_path = os.path.join(folder, fname.replace(".npy", ".tif"))
        tiff.imwrite(out_path, arr)
        print(f"Saved {out_path}")

print("✅ Done converting all .npy to .tif")