HMCNet

This repository provides the training and evaluation code for HMCNet, together with pretrained model weights and quantitative inference results.

Repository Structure

.
├── HMCNet/                         # HMCNet architecture and supporting modules
├── imagesTs/                       # Test images used by the provided evaluation example
├── labelsTs/                       # Corresponding test segmentation labels
├── results/                        # Excel files containing inference results
├── train.py                        # Training code
├── test.py                         # Evaluation code for AbdomenMRCT
├── requirements.txt                # Python dependencies
├── sub_abdomenMRCT_list.txt        # AbdomenMRCT data list
├── sub_abdomenMRCT_val_list.txt    # AbdomenMRCT validation/test image list
├── sub_abdomenMRCT_val_seg_list.txt # AbdomenMRCT validation/test label list
└── *.pth                           # Pretrained model weights managed with Git LFS

Data Preparation

Before training or testing, update the dataset paths in train.py, test.py, and the corresponding data-list files to match your local environment.

The provided imagesTs/ and labelsTs/ directories contain the image and segmentation-label structure used by the AbdomenMRCT evaluation example. Users should obtain all datasets from their original providers and comply with the corresponding licenses and data-use agreements.

Training

The complete training procedure is implemented in train.py.

After configuring the training data paths and other experiment settings, run:

python train.py

Training options, including dataset paths, checkpoint locations, and optimization settings, should be configured directly in train.py according to the target dataset and computing environment.

Testing on AbdomenMRCT

The provided test.py implements evaluation on the AbdomenMRCT dataset.

Before testing, verify the following settings in test.py:

The paths to the fixed and moving images.

The paths to the corresponding segmentation labels.

The path to the pretrained .pth checkpoint.

The output directory for registered images and evaluation results.

Run the evaluation with:

python test.py

Results

The results/ directory contains Excel files reporting the inference results of different registration models on six datasets:

OASIS

SLIVER

BTCV

AbdomenMRCT

MMWHS

ThoraxCBCT

The reported evaluation metrics include Dice score, average symmetric surface distance (ASSD), and the percentage of folding voxels (Folds), where available.

Citation

If you find this repository useful for your research, please cite the corresponding HMCNet paper. The complete citation information will be added upon publication.
