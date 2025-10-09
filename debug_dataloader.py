
import os
import numpy as np
import tifffile as tif
from monai.data import Dataset, DataLoader
from pathlib import Path
import json
from collections import defaultdict
import torch
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from train_tools.data_utils.transforms import train_transforms, valid_transforms, debug_train_transforms


# --- User-defined sampling ratios per dataset ---

sampling_ratios ={
    "01":0.05,
    "01_tiff":0.20,
    "BCCD_test_img":0.01,
    "BCCD_train_img":0.01, 
    "CoNIC_img":0.01,
    "DeepBacs_test_img":0.01,
    "DeepBacs_train_img":0.01,
    "IHC_TMA_img":0.01,
    "TNBC_img":0.01,
    "cellpose_img":0.14,
    "cellpose_test_img":0.15,
    "cpm15_img":0.01,
    "cpm17_test_img":0.01,
    "cpm17_train_img":0.01,
    "cyto2_img":0.20,
    "lynsec_img":0.01,
    "neurips_img":0.01,
    "nuinsseg_img":0.01,
    "panNuke2_img":0.01,
    "panNuke_img":0.01,
    "tissuenet_test_img":0.03,
    "tissuenet_train_img":0.03,
    "tissuenet_val_img":0.02,
    "yeast_BF_img":0.02,
    "yeast_PhC_img":0.01
}

# ---------------------------
# Dataset
# ---------------------------
class CustomMediarDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)

    def __getitem__(self, index):
        data = dict(self.data[index])  # Make a copy

        # Optional cellcenter
        cellcenter_path = data.get("cellcenter", None)
        if cellcenter_path:
            cellcenter_path = Path(cellcenter_path)
            if cellcenter_path.exists():
                data["cellcenter"] = str(cellcenter_path)
            else:
                print(f"[Warning] cellcenter file not found at index {index}: {cellcenter_path}")
        else:
            data.pop("cellcenter", None)

        # Optional precomputed flow
        flow_path = data.get("flow", None)
        if flow_path:
            flow_path = Path(flow_path)
            if flow_path.exists():
                flow_np = tif.imread(flow_path)
                data["flow"] = torch.from_numpy(flow_np).float()
            else:
                print(f"[Warning] flow file not found at index {index}: {flow_path}")
                data.pop("flow", None)
        else:
            data.pop("flow", None)

        if self.transform:
            data = self.transform(data)

        return data


# ---------------------------
# Load & decode mapping file
# ---------------------------
def path_decoder(root, mapping_file, no_label=False):
    data_dicts = []
    with open(mapping_file, "r") as file:
        data = json.load(file)

    for map_key in data.keys():
        for elem in data[map_key]:
            item = {"img": os.path.join(root, elem["img"])}
            if not no_label and "label" in elem:
                item["label"] = os.path.join(root, elem["label"])
            if "cellcenter" in elem:
                item["cellcenter"] = os.path.join(root, elem["cellcenter"])
            data_dicts.append(item)
    return data_dicts


# ---------------------------
# Group samples by dataset
# ---------------------------
def load_and_group_by_dataset(json_file):
    with open(json_file, "r") as f:
        mapping = json.load(f)

    grouped = defaultdict(list)
    for sample in mapping["official"]:
        dataset_name = sample["img"].split("/")[-2]
        grouped[dataset_name].append(sample)
    return grouped


# ---------------------------
# Custom weighted dataloader
# ---------------------------
def make_custom_dataloader(dataset, grouped, sampling_ratios, batch_size=4):
    total_ratio = sum(sampling_ratios.values())
    ratios = {k: v / total_ratio for k, v in sampling_ratios.items()}

    dataset_sizes = {k: len(v) for k, v in grouped.items()}

    weights = torch.zeros(len(dataset))
    for idx, sample in enumerate(dataset.data):
        dataset_name = sample["img"].split("/")[-2]
        weights[idx] = ratios[dataset_name] / dataset_sizes[dataset_name]

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


# ---------------------------
# Plotting helper
# ---------------------------
def show_image(img_tensor, title=None):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


# ---------------------------
# Main pipeline
# ---------------------------
root = "./"
mapping_file = "./train_tools/data_utils/mapping_labeled_dic_sim.json"

# 1. Decode all paths
all_data_dicts = path_decoder(root, mapping_file)

# 2. Group by dataset
grouped = load_and_group_by_dataset(mapping_file)
all_data = sum(grouped.values(), [])

# 3. Create dataset with transforms
trainset = CustomMediarDataset(all_data, transform=train_transforms)

# 4. Create weighted dataloader
train_loader = make_custom_dataloader(trainset, grouped, sampling_ratios, batch_size=1)


# 5. Visualize sampled images

max_images = 20
count = 0


for batch_idx, batch in enumerate(train_loader):
    images = batch['img']  # assuming shape [B, C, H, W]
    
    for i in range(len(images)):
        if count >= max_images:
            break
        
        show_image(images[i], title=f"Batch {batch_idx}, Sample {i}")
        count += 1
    
    if count >= max_images:
        break
    
    del batch



# import gc, tracemalloc, psutil
# for batch_idx, batch in enumerate(train_loader):
#     images = batch['img']
#     if batch_idx % 100 == 0:
#         mem = psutil.virtual_memory()
#         current, peak = tracemalloc.get_traced_memory()
#         print(f"[Iter {batch_idx}] CPU: {mem.used/1e9:.2f} GB | Python objects: {current/1e6:.1f} MB (peak {peak/1e6:.1f} MB)")
    
#     gc.collect()
#     del batch



# tracemalloc.start()
# print("starting")
# for i, batch in enumerate(dataloader):
#     #img = batch["img"].squeeze().cpu().numpy()


#     if i % 500 == 0:
#         mem = psutil.virtual_memory()
#         current, peak = tracemalloc.get_traced_memory()
#         print(f"[Iter {i}] CPU: {mem.used/1e9:.2f} GB | Python objects: {current/1e6:.1f} MB (peak {peak/1e6:.1f} MB)")
    
#     gc.collect()
#     del batch#, img


