import os
import numpy as np
import tifffile as tif
from skimage import io
from monai.data import Dataset, DataLoader
from pathlib import Path
import tifffile
import json
from collections import defaultdict
import psutil
import torch
from torch.utils.data import WeightedRandomSampler

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train_tools.data_utils.transforms import train_transforms, valid_transforms, debug_train_transforms


custom_ratios = {
    "BCCD_test_img": 0.03,
    "cellpose_img": 0.05,
    "CoNIC_img": 0.9,
    "cellpose_test_img": 0.02
}


class CustomMediarDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)

    def __getitem__(self, index):
        data = dict(self.data[index])  # Make a copy

        # Handle cellcenter paths as before
        cellcenter_path = data.get("cellcenter", None)
        if cellcenter_path is not None:
            cellcenter_path = Path(cellcenter_path)
            if not cellcenter_path.exists():
                print(f"[Warning] cellcenter file not found at index {index}: {cellcenter_path}")
            else:
                data["cellcenter"] = str(cellcenter_path)
        else:
            data.pop("cellcenter", None)

        # --- New: load precomputed flow ---
        flow_path = data.get("flow", None)
        if flow_path is not None:
            flow_path = Path(flow_path)
            if not flow_path.exists():
                print(f"[Warning] flow file not found at index {index}: {flow_path}")
                data.pop("flow", None)
            else:
                # Load flow data (assuming numpy .npy, adjust if different)
                flow_np = tifffile.imread(flow_path)
                flow_tensor = torch.from_numpy(flow_np).float()
                data["flow"] = flow_tensor
        else:
            data.pop("flow", None)
            
        if self.transform:
            data = self.transform(data)

        return data
    

def path_decoder(root, mapping_file, no_label=False, unlabeled=False):
    """Decode img/label/cellcenter file paths from root & mapping directory.

    Args:
        root (str): Base path for dataset
        mapping_file (str): JSON file containing image & label file paths
        no_label (bool): If True, do not include labels in output
        unlabeled (bool): If True, exclude certain corrupted images

    Returns:
        list[dict]: list of dictionaries (with keys "img", "label", optionally "cellcenter")
    """
    data_dicts = []

    with open(mapping_file, "r") as file:
        data = json.load(file)

        for map_key in data.keys():
            data_dict_item = []

            for elem in data[map_key]:
                item = {"img": os.path.join(root, elem["img"])}

                if not no_label and "label" in elem:
                    item["label"] = os.path.join(root, elem["label"])

                if "cellcenter" in elem:
                    item["cellcenter"] = os.path.join(root, elem["cellcenter"])

                data_dict_item.append(item)

            data_dicts += data_dict_item

    if unlabeled:
        data_dicts = [d for d in data_dicts if "00504" not in d["img"]]

    return data_dicts

def make_custom_dataloader(dataset, grouped, custom_ratios, batch_size=4):
    # Normalize ratios in case they don’t sum to 1
    total_ratio = sum(custom_ratios.values())
    ratios = {k: v / total_ratio for k,v in custom_ratios.items()}

    # Sizes per dataset
    dataset_sizes = {k: len(v) for k,v in grouped.items()}

    # Assign per-sample weights
    weights = torch.zeros(len(dataset))
    for idx, sample in enumerate(dataset.data):
        dataset_name = sample["img"].split("/")[-2]
        # Probability per sample = target_ratio / (#samples in that dataset)
        weights[idx] = ratios[dataset_name] / dataset_sizes[dataset_name]

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

def load_and_group_by_dataset(json_file):
    """Load mapping JSON and group samples by dataset name."""
    with open(json_file, "r") as f:
        mapping = json.load(f)

    # assumes your JSON has {"official": [...]} as in your script
    data_dicts = mapping["official"]

    grouped = defaultdict(list)
    for sample in data_dicts:
        # dataset key: extract from path
        # (here: parent folder name of the image dir, adapt if needed)
        dataset_name = sample["img"].split("/")[-2]  
        grouped[dataset_name].append(sample)

    return grouped

def get_dataloaders_labeled(
    root,
    mapping_file,
    batch_size=1,
    sampling_ratios=False
):
    """Set DataLoaders for labeled datasets.

    Args:
        root (str): root directory
        mapping_file (str): json file for mapping dataset
        valid_portion (float, optional): portion of valid datasets. Defaults to 0.1.
        batch_size (int, optional): batch size. Defaults to 8.
        shuffle (bool, optional): shuffles dataloader. Defaults to True.
        num_workers (int, optional): number of workers for each datalaoder. Defaults to 5.

    Returns:
        dict: dictionary of data loaders.
    """
    if(sampling_ratios):
        grouped = load_and_group_by_dataset(mapping_file)
        all_data = sum(grouped.values(), [])
        trainset = CustomMediarDataset(all_data, transform=train_transforms)
        train_loader = make_custom_dataloader(trainset, grouped, custom_ratios, batch_size=batch_size)
        return train_loader, all_data

    # Get list of data dictionaries from decoded paths
    data_dicts = path_decoder(root, mapping_file)


    # Obtain datasets with transforms
    trainset = CustomMediarDataset(data_dicts, transform=train_transforms)

    # Set dataloader for Trainset
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, data_dicts

dataloader, data_dict = get_dataloaders_labeled(
    root="./",
    mapping_file="./train_tools/data_utils/mapping_labeled_dic_sim.json",
    batch_size=1,
)

all_data = sum(grouped.values(), [])

trainset = CustomMediarDataset(all_data, transform=train_transforms)

train_loader = make_custom_dataloader(trainset, grouped, custom_ratios, batch_size=4)


import matplotlib.pyplot as plt
import numpy as np

# Function to plot a single image tensor
def show_image(img_tensor, title=None):
    # img_tensor: [C,H,W], convert to HWC
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # If normalized between 0-1 or mean/std applied, scale to 0-1 for plotting
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# # Iterate over train_loader
# for batch_idx, batch in enumerate(train_loader):
#     images = batch['img']  # [B,C,H,W]
#     labels = batch.get('label', None)
    
#     for i in range(images.shape[0]):
#         title = None
#         if labels is not None:
#             title = str(labels[i])  # optionally show label path
#         show_image(labels[i], title=title)

max_images = 20
count = 0
for batch_idx, batch in enumerate(train_loader['train']):
    images = batch['img']
    for i in range(images.shape[0]):
        show_image(images[i])
        count += 1
        if count >= max_images:
            break
    if count >= max_images:
        break


# import gc, tracemalloc

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


