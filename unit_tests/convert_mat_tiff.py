import os
import numpy as np
import tifffile as tiff
from scipy.io import loadmat
import h5py

folder = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/TNBC_masks"

def load_mat_file(path, key=None):
    """Load .mat file and return array."""
    try:
        mat = loadmat(path)
        if key is None:
            # pick first non-metadata key
            key = [k for k in mat.keys() if not k.startswith("__")][0]
        return mat[key]
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            if key is None:
                key = list(f.keys())[0]
            return np.array(f[key])

for fname in os.listdir(folder):
    if fname.endswith(".mat"):
        fpath = os.path.join(folder, fname)
        arr = load_mat_file(fpath)

        # squeeze singleton dims
        arr = np.squeeze(arr)

        # save as tif
        out_path = os.path.join(folder, fname.replace(".mat", ".tif"))
        tiff.imwrite(out_path, arr.astype(np.uint16))

        print(f"Saved {out_path}")

print("✅ Done converting all .mat to .tif")