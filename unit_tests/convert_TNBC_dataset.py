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


get_image_shapes("/hpcwork/cs088267/scratch/cellpose_pretraining_data/TNBC_masks")

# import os
# from PIL import Image

# def convert_to_rgb(input_dir, output_dir=None):
#     """
#     Converts all PNG images in a directory to 3-channel RGB if they have 4 channels.
#     Saves them in the same directory unless output_dir is specified.
#     """
#     if output_dir is None:
#         output_dir = input_dir
#     os.makedirs(output_dir, exist_ok=True)

#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith(".png"):
#             filepath = os.path.join(input_dir, filename)
#             with Image.open(filepath) as img:
#                 # Check if image has an alpha channel (RGBA)
#                 if img.mode == "RGBA":
#                     img = img.convert("RGB")
#                     print(f"Converted {filename} to RGB.")
#                 else:
#                     print(f"Skipped {filename}, already {img.mode}.")
#                 img.save(os.path.join(output_dir, filename))

# if __name__ == "__main__":
#     input_directory = "/hpcwork/cs088267/scratch/cellpose_pretraining_data/TNBC_img"
#     output_directory = None  # or set a path like "path/to/output"
#     convert_to_rgb(input_directory, output_directory)