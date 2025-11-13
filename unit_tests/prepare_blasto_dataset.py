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
    # 🔧 Input: root directory containing subfolders (each has "images" and "masks")
    root_dir = "/work/scratch/geiger/Datasets/Blasto/final_test"
    buffer = 50

    # 🔧 Output directories (global)
    out_img_dir = os.path.join(root_dir, "cropped_images")
    out_mask_dir = os.path.join(root_dir, "cropped_masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Iterate through all subfolders in root_dir
    for sample_name in os.listdir(root_dir):
        sample_path = os.path.join(root_dir, sample_name)
        if not os.path.isdir(sample_path):
            continue

        img_dir = os.path.join(sample_path, "images")
        mask_dir = os.path.join(sample_path, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"⚠️ Skipping {sample_name}: missing 'images/' or 'masks/' folder.")
            continue

        # Find all .npy files in images folder
        image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".npy")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".npy")])

        # Match image and mask by index
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            try:
                img = np.load(img_path)
                mask = np.load(mask_path)

                cropped_img, cropped_mask, coords = crop_to_mask_with_buffer(img, mask, buffer)

                base_name = f"{sample_name}_{os.path.splitext(img_file)[0].replace('image_', '')}"
                out_img_path = os.path.join(out_img_dir, f"{base_name}_cropped.tif")
                out_mask_path = os.path.join(out_mask_dir, f"{base_name}_cropped.tif")

                tif.imwrite(out_img_path, cropped_img.astype(img.dtype))
                tif.imwrite(out_mask_path, cropped_mask.astype(mask.dtype))

                print(f"✅ Saved cropped {sample_name}/{img_file} → {base_name}_cropped.tif")
                print("   Crop coords:", coords)

            except Exception as e:
                print(f"❌ Error processing {sample_name}/{img_file}: {e}")

    print("\n🎉 Done! All cropped images saved to:")
    print("  -", out_img_dir)
    print("  -", out_mask_dir)