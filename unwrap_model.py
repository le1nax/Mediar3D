import torch
from collections import OrderedDict
import os

def unwrap_ddp_checkpoint(input_path, output_path=None):
    """
    Load a DDP-wrapped checkpoint and save it without 'module.' prefixes.

    Args:
        input_path (str): Path to the checkpoint file (.pth) saved from a DDP model.
        output_path (str, optional): Where to save the cleaned checkpoint.
                                    If None, a new file with '_unwrapped' suffix is created.

    Returns:
        str: Path to the saved cleaned checkpoint.
    """
    state_dict = torch.load(input_path, map_location="cpu")

    # remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    # decide output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_unwrapped{ext}"

    torch.save(new_state_dict, output_path)
    return output_path


clean_path = unwrap_ddp_checkpoint("../../W_B/Hiera_PT_hpc/pretrained_hiera_01val_4gpus_16bs_2025-10-07_07-38-17.pth")
print(f"Saved cleaned model to {clean_path}")