# HMCNet

This repository provides the training and evaluation code for HMCNet, together with pretrained model weights and quantitative inference results.

## Repository Structure

```text
.
├── HMCNet/                          # HMCNet architecture and supporting modules
├── imagesTs/                        # Test images for the evaluation example
├── labelsTs/                        # Corresponding segmentation labels
├── results/                         # Excel files containing inference results
├── train.py                         # Training code
├── test.py                          # Evaluation code for AbdomenMRCT
├── requirements.txt                 # Python dependencies
├── sub_abdomenMRCT_list.txt         # AbdomenMRCT data list
├── sub_abdomenMRCT_val_list.txt     # AbdomenMRCT image list
├── sub_abdomenMRCT_val_seg_list.txt # AbdomenMRCT segmentation-label list
└── *.pth                            # Pretrained weights managed with Git LFS
```

## Data Preparation

Before training or testing, update the dataset paths in `train.py`, `test.py`, and the corresponding data-list files according to your local environment.

The provided `imagesTs/` and `labelsTs/` directories follow the data structure used by the AbdomenMRCT evaluation example.

## Training

The complete training procedure is implemented in `train.py`.

After configuring the training data paths, output directories, and other experimental settings, run:

```bash
python train.py
```

The training options, including dataset paths, checkpoint locations, and optimization settings, can be configured directly in `train.py`.

## Testing on AbdomenMRCT

The provided `test.py` implements evaluation on the AbdomenMRCT dataset.

Before running the evaluation, verify the following settings in `test.py`:

1. Paths to the fixed and moving images.
2. Paths to the corresponding segmentation labels.
3. Path to the pretrained `.pth` checkpoint.
4. Output directory for registered images and evaluation results.

Run the evaluation using:

```bash
python test.py
```

## Results

The `results/` directory contains Excel files reporting the inference results of different registration models on six datasets:

- OASIS
- SLIVER
- BTCV
- AbdomenMRCT
- MMWHS
- ThoraxCBCT

The reported evaluation metrics include:

- Dice score;
- average symmetric surface distance (ASSD);
- percentage of folding voxels (Folds), where available.


## Citation

If you find this repository useful for your research, please cite the corresponding HMCNet paper. The complete citation information will be added upon publication.
