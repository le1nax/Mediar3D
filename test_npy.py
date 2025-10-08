import numpy as np

img_path = "/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/BlastoSPIM1_train/M8_020/crop/M8_020_crop_0001.npy"
im = np.load(img_path, allow_pickle=True)

print("Type:", type(im))
print("Shape:", getattr(im, "shape", None))
print("Dtype:", getattr(im, "dtype", None))
print("Value:", im)