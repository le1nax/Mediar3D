import torch
import argparse, pprint, os, psutil, sys
import numpy as np

# add repo root (one level up) to sys.path so train_tools is found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train_tools import *
from SetupDict import MODELS, PREDICTOR


def postprocess_shifted_windows(pred, predictor, window_size=(128, 128, 128), overlap=0.25):
    """
    Apply _post_process3D in a shifted window manner.
    
    pred: 4D NumPy array (C, Z, Y, X)
    predictor: initialized predictor with _post_process3D
    window_size: tuple of (wZ, wY, wX)
    overlap: fraction of overlap between windows
    """
    C, Z, Y, X = pred.shape
    wz, wy, wx = window_size
    sz, sy, sx = int(wz * (1 - overlap)), int(wy * (1 - overlap)), int(wx * (1 - overlap))

    # Prepare final mask array
    mask_final = np.zeros((Z, Y, X), dtype=np.uint32)
    count_mask = np.zeros((Z, Y, X), dtype=np.float32)  # for averaging overlaps

    # Generate grid coordinates
    z_starts = list(range(0, Z, sz))
    y_starts = list(range(0, Y, sy))
    x_starts = list(range(0, X, sx))

    for z0 in z_starts:
        z1 = min(z0 + wz, Z)
        for y0 in y_starts:
            y1 = min(y0 + wy, Y)
            for x0 in x_starts:
                x1 = min(x0 + wx, X)

                # Extract window
                pred_win = pred[:, z0:z1, y0:y1, x0:x1]

                # Run postprocessing
                mask_win = predictor._post_process3D(pred_win)

                # Accumulate into final mask
                mask_final[z0:z1, y0:y1, x0:x1] += mask_win
                count_mask[z0:z1, y0:y1, x0:x1] += 1

    # Average overlapping regions
    mask_final = (mask_final / np.maximum(count_mask, 1)).astype(np.uint32)
    return mask_final


def main(args):
    """Init predictor and run shifted-window postprocess on fake tensor"""

    # -------------------------------
    # 1. Initialize model and predictor
    # -------------------------------
    model_args = args.pred_setups.model
    model = MODELS[model_args.name](**model_args.params)

    # load weights (CPU for testing)
    weights = torch.load(args.pred_setups.model_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)

    predictor = PREDICTOR[args.pred_setups.name](
        model,
        args.pred_setups.device,
        args.pred_setups.input_path,
        args.pred_setups.output_path,
        args.pred_setups.cellcenters_path,
        args.pred_setups.make_submission,
        args.pred_setups.exp_name,
        args.pred_setups.algo_params,
    )

    # -------------------------------
    # 2. Create fake prediction tensor
    # -------------------------------
    Z, Y, X = 256, 256, 256  # adjust for testing
    pred = np.random.rand(4, Z, Y, X).astype(np.float32)

    print(f"[DEBUG] Fake prediction created: {pred.shape}")
    print("Tensor size (GB):", pred.nbytes / 1024**3)

    process = psutil.Process(os.getpid())
    print("Memory before postprocess: %.2f GB" % (process.memory_info().rss / 1024**3))

    # -------------------------------
    # 3. Run shifted-window postprocessing
    # -------------------------------
    try:
        masks = predictor._post_process3D(pred)
        print("[DEBUG] Shifted-window postprocess succeeded. Masks shape:", masks.shape)
    except RuntimeError as e:
        print("[DEBUG] Postprocess failed with RuntimeError:", e)
    except Exception as e:
        print("[DEBUG] Postprocess failed with Exception:", e)

    print("Memory after postprocess: %.2f GB" % (process.memory_info().rss / 1024**3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file processing")
    parser.add_argument(
        "--config_path", default="./config/step3_prediction/base_prediction.json", type=str
    )
    args = parser.parse_args()

    opt = ConfLoader(args.config_path).opt
    pprint_config(opt)

    main(opt)