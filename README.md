
# What this repo is about


This repository forks the original [MEDIAR](https://arxiv.org/abs/2212.03465) for 3D inference.
Main contribution:
1. Memory friendly, fast 3D Inference
2. New training pipeline for incomplete annotations
3. Ablation study using larger pretraining dataset and SAM2 Encoder (currently in a [seperate repository](https://github.com/le1nax/Mediar3D_SAM2Ablation))
4. Experiments with new pretrained Mediar models on various 3D datasets, benchmarking Mediar and ablations against CellposeSAM

The developed models perform similar on various 3D datasets while being significantly faster, but depended on a post-processing parameter (Cell probability threshold).

# How to setup the environment 


1. Download [miniforge](https://github.com/conda-forge/miniforge)
2. Clone this repository and navigate to the repository directory
3. Source miniforge and recreate environment using the `environment_full.yml` file (conda env create -f environment.yml)


# How to perform inference


Configure json config `step3_prediction/base_prediciton.json` and run `predict.py`. Providing `cellcenters_path` will post-process the predictions with given detection masks.


# How to prepare your data for training


Use the script `Dataloading/gererate_mapping_multiple_datasets.py` and edit the lines for image and mask locations. This will create the json mapping file.

# How to pretrain with new pretraining dataset


Use branch multigpu to run with multiple gpus. If you have access to the intern filesystem, use datasets: /netshares/BiomedicalImageAnalysis/Resources/dataset_collection/cellpose_pretraining_data and set sampling probabilities for each dataset in `step1_pretraining/phase1.json`.


# How to finetune your model


Configure `step2_finetuning/finetuning1.json` 
Enabling `incomplete_annotations` will mask the loss to regions of interest + some dilated pixels.
Setting `save_at_rois` will checkpoint the model at given amounts of ROIs seen in training.
If validation fraction is given, the validation data will be used for checkpointing at valid epochs, which occur at a given validation frequency.
