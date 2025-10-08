
import numpy as np
import tifffile as tif
import os

def crop_to_mask_with_buffer(image: np.ndarray, mask: np.ndarray, buffer: int = 50):
    """
    Crop a 3D image and mask to the ROI where mask is nonzero, with buffer.
    No padding outside the image boundaries.
    """
    assert image.shape == mask.shape, "Image and mask must have same shape"
    assert image.ndim == 3, "Function assumes 3D volumes"

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty, no ROI found")

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min = max(z_min - buffer, 0)
    y_min = max(y_min - buffer, 0)
    x_min = max(x_min - buffer, 0)

    z_max = min(z_max + buffer, image.shape[0] - 1)
    y_max = min(y_max + buffer, image.shape[1] - 1)
    x_max = min(x_max + buffer, image.shape[2] - 1)

    cropped_image = image[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

    return cropped_image, cropped_mask, ((z_min, z_max), (y_min, y_max), (x_min, x_max))


if __name__ == "__main__":
    # 🔧 Set your .npy input paths
    image_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/BlastoSPIM1_train/F1_095/images/F1_095_image_0001.npy"
    mask_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/BlastoSPIM1_train/F1_095/masks/F1_095_masks_0001.npy"
    out_dir = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/BlastoSPIM1_train/F1_095/cropped"
    buffer = 50

    # Load NumPy volumes
    img = np.load(image_path)
    mask = np.load(mask_path)

    cropped_img, cropped_mask, coords = crop_to_mask_with_buffer(img, mask, buffer)

    os.makedirs(out_dir, exist_ok=True)

    # Save cropped volumes as TIFF
    tif.imwrite(os.path.join(out_dir, "cropped_image.tif"), cropped_img.astype(img.dtype))
    tif.imwrite(os.path.join(out_dir, "cropped_mask.tif"), cropped_mask.astype(mask.dtype))

    print("Saved cropped TIFFs to", out_dir)
    print("Crop coords:", coords)
