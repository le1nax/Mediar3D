import numpy as np
import tifffile as tiff
import os

# load TissueNet split
npz_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_v1.1/tissuenet_v1.1_test.npz"  # change to train/val as needed
data = np.load(npz_path)

X, y = data["X"], data["y"]  # X: images, y: labels

# check shapes
print("X shape:", X.shape)  # (N, H, W, C)
print("y shape:", y.shape)  # (N, H, W, 2)

# output dirs
out_imgs = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_images_test"
out_labels = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_labels_test"
os.makedirs(out_imgs, exist_ok=True)
os.makedirs(out_labels, exist_ok=True)

for i, (img, mask) in enumerate(zip(X, y)):
    img_name = f"cell_{i:05d}.tif"
    mask_name_nuc = f"cell_{i:05d}_label.tif"

    # take one channel of X (e.g. channel 0)
    single_img = img[..., 0]   # shape (256, 256)

    # save image
    tiff.imwrite(os.path.join(out_imgs, img_name), single_img.astype(np.uint8))

    # save masks separately, each (256, 256)
    tiff.imwrite(os.path.join(out_labels, mask_name_nuc), mask[..., 1].astype(np.uint16))


#     import numpy as np
# import tifffile as tiff
# import os

# # paths
# base_dir = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_v1.1"
# out_imgs = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_images"
# out_labels = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/tissuenet_labels"

# os.makedirs(out_imgs, exist_ok=True)
# os.makedirs(out_labels, exist_ok=True)

# # npz files
# splits = [
#     "tissuenet_v1.1_train.npz",
#     "tissuenet_v1.1_val.npz",
#     "tissuenet_v1.1_test.npz",
# ]

# counter = 0
# for split in splits:
#     npz_path = os.path.join(base_dir, split)
#     print(f"Processing {npz_path} ...")

#     data = np.load(npz_path)
#     X, y = data["X"], data["y"]

#     print("  X shape:", X.shape)
#     print("  y shape:", y.shape)

#     for img, mask in zip(X, y):
#         img_name = f"cell_{counter:05d}.tif"
#         mask_name = f"cell_{counter:05d}_label.tif"

#         # use first channel of X (grayscale)
#         single_img = img[..., 0]  # shape (256, 256)

#         # save image + mask
#         tiff.imwrite(os.path.join(out_imgs, img_name), single_img.astype(np.uint8))
#         tiff.imwrite(os.path.join(out_labels, mask_name), mask[..., 1].astype(np.uint16))

#         counter += 1

# print(f"✅ Done! Exported {counter} images and masks.")