#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import swanlab
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from thop import profile
import statistics
import copy
import psutil
import pandas as pd

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8




# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', default="./sub_abdomenMRCT_list.txt",help='line-seperated list of training files')
parser.add_argument('--seg-atlas', default="./images/OASIS_OAS1_0388_MR1/aligned_norm.nii.gz",
                    help='seg-atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--val-t1-list', default="./sub_val_list.txt",help='line-seperated list of val files')
parser.add_argument('--val-seg-list', default="./sub_val_seg_list.txt",help='line-seperated list of val files')

parser.add_argument('--train-seg-list', default="./sub_abdomenMRCT_list.txt",help='line-seperated list of val files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')

parser.add_argument('--model-dir', default='./result',
                    help='model output directory (default: models)')

parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--exp-name', default='dualVxm',  # 
                                    help='Experiment name')  # 保存log和模型对应的实验设置的名字
parser.add_argument('--notes', help='notes for wandb')  # 保存log和模型对应的实验设置的名字
                    
# training parameters
parser.add_argument('--gpu', default='2', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# multi_resolution parameters
parser.add_argument('--layer1-epochs', type=int, default=200,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer2-epochs', type=int, default=300,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer3-epochs', type=int, default=400,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer4-epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--layer5-epochs', type=int, default=1000,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer2', type=int, default=0,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer3', type=int, default=0,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer4', type=int, default=0,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--freeze-layer5', type=int, default=0,
                    help='number of training epochs (default: 1500)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image_loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=1,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()


bidir = args.bidir

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# # 加载验证集的数据

val_t1_files = vxm.py.utils.read_file_list(args.val_t1_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
val_seg_files = vxm.py.utils.read_file_list(args.val_seg_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)                                       
assert len(val_t1_files) > 0, 'Could not find any val data.'

atlas,atlas_affine = vxm.py.utils.load_volfile("./images/OASIS_OAS1_0388_MR1/aligned_norm.nii.gz", np_var='vol',
                                  add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
seg_atlas = vxm.py.utils.load_volfile(args.seg_atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
val_t1_generator = vxm.generators.seg_scan_to_atlas(val_t1_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
val_seg_generator = vxm.generators.seg_scan_to_atlas(val_seg_files, seg_atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = (160,192,224)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# assert np.mod(args.batch_size, nb_gpus) == 0, \
#     'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet



# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
elif args.image_loss == 'MI':
    image_loss_func = vxm.losses.MutualInformation()
elif args.image_loss == 'MIND':
    image_loss_func = vxm.losses.MINDLoss().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
# losses += [vxm.losses.Grad3d(penalty='l2')]
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

loss_mse = [vxm.losses.MSE().loss]

config = dict(
    threshold=0.4)

def set_random_seed(seed):
    """
    设置随机种子，以确保实验结果的可重复性。
    """
    random.seed(seed)  # 设置 Python random 模块的种子
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
    torch.backends.cudnn.deterministic = True  # 确保 CuDNN 给出确定性结果
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 自动调优（对固定输入尺寸有效）


def test():
    print("test")

    #  #    # ###########  HMCNet ##################################
    # model = vxm.networks.dual_pyramid_VxmDense_doubleTrans_AMFFM_iterative2221(inshape=inshape)
    # checkpoint = torch.load("./best_model_dual_pyramid_VxmDense_doubleTrans_AMFFM_iterative2221.pth")
    # model.to(device)
    # model_dict = model.state_dict()
    # state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    # model.eval()

     ###########  HMCNet-diff ##################################
    model = vxm.networks.dual_pyramid_VxmDense_doubleTrans_AMFFM_iterative2221_diff(inshape=inshape)
    checkpoint = torch.load("./best_model_dual_pyramid_VxmDense_doubleTrans_AMFFM_iterative2221_diff.pth")
    model.to(device)
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    #  ------------------------------------------------------------------------------------
    # Calculate FLOPs and Parameters:vm,autofuse
    dummy_input = torch.randn(1, 1, 160, 192, 224).to(device)
    flops, params = profile(model, (dummy_input, dummy_input,))
    flops_m = flops / 1000000.0
    params_m = params / 1000000.0
    print('flops: %.2f M, params: %.2f M' % (flops_m, params_m))


    # ################# PDFNet-trans ################################
    case_results = []
    inference_times = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    num = 0

    
    print("--------- Validation Dice ---------")
    for step_outer in range(len(val_t1_files)):
        # Extract moving data
        inputs_moving, y_true = next(val_t1_generator)
        inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
        
        # Extract moving seg data
        seg_inputs_moving, seg_y_true = next(val_seg_generator)
        seg_inputs_moving = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_moving]
        
        for step_inner in range(len(val_t1_files)):
            # Extract fixed data
            inputs_fixed, y_true = next(val_t1_generator)
            inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
            
            # Extract fixed seg data
            seg_inputs_fixed, seg_y_true = next(val_seg_generator)
            seg_inputs_fixed = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
            
            if step_inner < len(val_t1_files) - 1:
                with torch.no_grad():
                    starter.record()
            
                    # HMCnet
                    flow_1, flow_2, flow_3, flow = model(inputs_moving[0], inputs_fixed[0])

                    
                    ender.record()
                    torch.cuda.synchronize() # Synchronize GPU time
                    
                    curr_time = starter.elapsed_time(ender)
                    print(curr_time)
                    inference_times.append(curr_time)
                    
                    # Generate moved image
                    SPT = vxm.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device)
                    seg_pre = SPT(seg_inputs_moving[0], flow)
                    
                    # Calculate Dice
                    dice, dice_region = vxm.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0],dim=36)
                    # dice, dice_region = vxm.torch.losses.compute_dice(seg_inputs_moving[0], seg_inputs_fixed[0])
                    
                    # Calculate ASSD
                    # assd = vxm.torch.losses.compute_ASSD(seg_inputs_fixed[0].cpu(), seg_inputs_moving[0].cpu())
                    assd = vxm.torch.losses.compute_ASSD(seg_inputs_fixed[0].cpu(), seg_pre.cpu(),dim=36)
                    # assd = 1
                    
                    # Calculate Jacobian determinant
                    filed = torch.zeros([160, 192, 224, 3]).to(device)
                    filed[:, :, :, 0] = flow[0, 0, :, :, :]
                    filed[:, :, :, 1] = flow[0, 1, :, :, :]
                    filed[:, :, :, 2] = flow[0, 2, :, :, :]
                    jac_det = vxm.torch.losses.jacobian_determinant(filed.cpu())
                    Jd = np.sum(jac_det < 0)
                    # Jd = 0

                    
                    # Ensure metrics are standard python floats for DataFrame storage
                    dice_val = dice.mean().item() if isinstance(dice, torch.Tensor) else dice
                    assd_val = assd.item() if isinstance(assd, torch.Tensor) else assd
                    
                    if dice_val < 1:
                        print(f"Case {num} - Dice: {dice_val:.4f}, ASSD: {assd_val:.4f}, JD: {Jd}")
                        case_results.append({
                            'Case': num,
                            'Dice': dice_val,
                            'ASSD': assd_val,
                            'JD': Jd
                        })
                        num += 1

        print("---------- Change fixed image ---------")

    # Convert case results to DataFrame
    df_cases = pd.DataFrame(case_results)

    # Calculate summary metrics (Means and Standard Deviations)
    # Using ddof=1 (default in pandas) for sample standard deviation
    if not df_cases.empty:
        mean_dice = df_cases['Dice'].mean()
        std_dice = df_cases['Dice'].std()
        
        mean_assd = df_cases['ASSD'].mean()
        std_assd = df_cases['ASSD'].std()
        
        mean_jd = df_cases['JD'].mean()
        std_jd = df_cases['JD'].std()
    else:
        mean_dice = std_dice = mean_assd = std_assd = mean_jd = std_jd = 0

    mean_time = np.mean(inference_times) if inference_times else 0

    print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Mean ASSD: {mean_assd:.4f} ± {std_assd:.4f}")
    print(f"Mean JD: {mean_jd:.4f} ± {std_jd:.4f}")
    print(f"Mean Inference Time (ms): {mean_time:.4f}")




if __name__ == "__main__":
    test()



    





    