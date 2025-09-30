import os
import sys
import numpy as np
from PIL import Image

def get_image_shapes(directory):
    """
    Reads all PNG images in the directory and prints their shapes.
    Shape format: (height, width, channels)
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return

    files = [f for f in os.listdir(directory) if f.lower().endswith(".png")]
    if not files:
        print("No PNG files found in the directory.")
        return

    for file in files:
        path = os.path.join(directory, file)
        try:
            img = Image.open(path)
            arr = np.array(img)
            print(f"{file}: {arr.shape}")
        except Exception as e:
            print(f"Could not read {file}: {e}")



get_image_shapes("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data/nuinsseg_img")
