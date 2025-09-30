import json
import os
from PIL import Image

# --- CONFIG ---
mapping_json = "/work/scratch/geiger/repos/MEDIAR/train_tools/data_utils/mapping_labeled_dic_sim.json"

# Load mapping
with open(mapping_json, "r") as f:
    mapping = json.load(f)

def check_image(path):
    """Return number of channels if PNG, otherwise None."""
    if not path.lower().endswith(".png"):
        return None  # ignore non-PNG
    if not os.path.exists(path):
        return "MISSING"
    try:
        with Image.open(path) as img:
            return len(img.getbands())  # e.g. ('R','G','B','A') -> 4
    except Exception as e:
        return f"ERROR: {e}"

# Iterate and print only 4-channel PNGs
for i, entry in enumerate(mapping.get("official", [])):
    for key in ["img", "label"]:
        path = entry[key]
        result = check_image(path)
        if result == 4:
            print(f"[{i}] {key}: {path} -> 4 channels")