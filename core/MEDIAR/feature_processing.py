import numpy as np
from skimage.measure import regionprops, label
from skimage import measure

# def fill_holes_and_remove_small_masks(pred_mask):
#     pred_mask = pred_mask > 0.5
#     pred_mask = morphology.remove_small_holes(pred_mask, connectivity=1)
#     pred_mask = morphology.remove_small_objects(pred_mask, 16)
#     pred_mask = measure.label(pred_mask)
#     return pred_mask