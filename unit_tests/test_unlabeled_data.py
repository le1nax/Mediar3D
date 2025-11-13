import os
import numpy as np
import tifffile as tiff
import random

def reduce_annotations(
    label_dir: str,
    output_dir: str = None,
    keep_ratio: float = 0.4,
    seed: int = 42,
):
    """
    Randomly removes a portion of instance masks from label images.

    Parameters
    ----------
    label_dir : str
        Directory containing label TIFF images.
    output_dir : str, optional
        Where to save reduced labels (default: same as input).
    keep_ratio : float, optional
        Fraction of instances to keep (default: 0.2 = remove 80%).
    seed : int, optional
        Random seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(label_dir, "reduced_labels")
    os.makedirs(output_dir, exist_ok=True)

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".tif") or f.endswith(".tiff")]

    for f in label_files:
        label_path = os.path.join(label_dir, f)
        label_img = tiff.imread(label_path)

        unique_ids = np.unique(label_img)
        unique_ids = unique_ids[unique_ids != 0]  # exclude background

        if len(unique_ids) == 0:
            print(f"[!] Skipping {f}: no labeled instances found.")
            continue

        n_keep = max(1, int(len(unique_ids) * keep_ratio))
        kept_ids = set(random.sample(list(unique_ids), n_keep))

        # Create reduced label image
        reduced_img = np.where(np.isin(label_img, list(kept_ids)), label_img, 0)

        # Save
        out_path = os.path.join(output_dir, f)
        tiff.imwrite(out_path, reduced_img.astype(label_img.dtype))

        print(f"[✓] Processed {f}: kept {n_keep}/{len(unique_ids)} instances → saved to {out_path}")

    print("\n✅ Done. Reduced labels saved in:", output_dir)


if __name__ == "__main__":
    # Example usage
    label_dir = "/work/scratch/geiger/Datasets/Blasto/train/F1_095_patially_annoated/masks" 
    out = "/work/scratch/geiger/Datasets/Blasto/train/F1_095_patially_annoated/partially_annotated_masks" 
    reduce_annotations(label_dir, output_dir=out)