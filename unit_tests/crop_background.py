import tifffile as tiff
import numpy as np
import os

def crop_and_save_slices(input_path, img_out_dir, mask_out_dir, thresh_x, thresh_y):
    """
    Crop a subvolume from a 3D image stack and save each z-slice separately.
    Also saves empty masks with the same shape in another directory.
    
    input_path: path to 3D TIFF [Z, Y, X]
    img_out_dir: directory for cropped z-slices
    mask_out_dir: directory for corresponding empty masks
    thresh_x, thresh_y: crop thresholds (inclusive)
    """
    # Load 3D image
    img = tiff.imread(input_path)  # [Z, Y, X]
    print(f"Loaded image shape: {img.shape}")

    # Crop subvolume
    cropped = img[:, :thresh_y+1, :thresh_x+1]  # [Z, newY, newX]
    print(f"Cropped subvolume shape: {cropped.shape}")

    # Create output directories
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # Save slices
    for z in range(cropped.shape[0]):
        img_slice = cropped[z]
        mask_slice = np.zeros_like(img_slice, dtype=np.uint8)

        img_filename = os.path.join(img_out_dir, f"slice_{z:04d}.tiff")
        mask_filename = os.path.join(mask_out_dir, f"slice_{z:04d}_label.tiff")

        tiff.imwrite(img_filename, img_slice.astype(img.dtype))
        tiff.imwrite(mask_filename, mask_slice)

    print(f"Saved {cropped.shape[0]} slices to {img_out_dir} and {mask_out_dir}")


if __name__ == "__main__":
    input_path = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test/t000.tiff"
    img_out_dir = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test_images_background"
    mask_out_dir = "/work/scratch/geiger/Datasets/CTC/Fluo-N3DL-TRIF/02_test_labels_background"

    thresh_x = 88
    thresh_y = 88

    crop_and_save_slices(input_path, img_out_dir, mask_out_dir, thresh_x, thresh_y)