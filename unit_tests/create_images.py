import os
import numpy as np
import tifffile as tiff

# --- Configuration ---
input_path = "/home/students/geiger/Downloads/dummy_image.tif"  # Path to your input 3D TIFF
output_dir = "/work/scratch/geiger/Datasets/testing/img"
sizes = [800]  # target cube sizes

os.makedirs(output_dir, exist_ok=True)


# --- Load 3D image ---
volume = tiff.imread(input_path)  # shape: (Z, Y, X)
print(f"Loaded volume shape: {volume.shape}")

# --- Helper: extract centered cube with mirror padding ---
def extract_centered_cube(vol, size):
    z, y, x = vol.shape
    cz, cy, cx = z // 2, y // 2, x // 2
    hz = size // 2
    hy = size // 2
    hx = size // 2

    # Compute start/end indices
    z1, z2 = cz - hz, cz + hz
    y1, y2 = cy - hy, cy + hy
    x1, x2 = cx - hx, cx + hx

    # Clip to image bounds
    z1_clip, z2_clip = max(z1, 0), min(z2, z)
    y1_clip, y2_clip = max(y1, 0), min(y2, y)
    x1_clip, x2_clip = max(x1, 0), min(x2, x)

    cube = vol[z1_clip:z2_clip, y1_clip:y2_clip, x1_clip:x2_clip]

    # Determine how much padding is needed in each direction
    pad_before_z = max(0, -z1)
    pad_after_z = max(0, z2 - z)
    pad_before_y = max(0, -y1)
    pad_after_y = max(0, y2 - y)
    pad_before_x = max(0, -x1)
    pad_after_x = max(0, x2 - x)

    if any(p > 0 for p in [pad_before_z, pad_after_z, pad_before_y, pad_after_y, pad_before_x, pad_after_x]):
        cube = np.pad(
            cube,
            (
                (pad_before_z, pad_after_z),
                (pad_before_y, pad_after_y),
                (pad_before_x, pad_after_x),
            ),
            mode="reflect",
        )

    return cube

# --- Extract and save ---
for s in sizes:
    cube = extract_centered_cube(volume, s)
    out_path = os.path.join(output_dir, f"cube_{s}x{s}x{s}.tif")
    tiff.imwrite(out_path, cube.astype(volume.dtype))
    print(f"Saved: {out_path} (shape={cube.shape})")

print("Done.")