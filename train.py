import os
import random
import swanlab
import argparse
import time
import math
import numpy as np
import torch
import gc
import torch.nn as nn
import wandb
# from patchify import patchify,unpatchify
import sys
import pandas as pd
from torchvision import transforms
import glob
from torch.utils.data import DataLoader
import ml_collections
from thop import profile
import statistics
import copy
import psutil



# sys.path.append('/home/boys/project/XMorpher')
# from data import datasets, trans

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import HMCNet as HMCNet  # nopep8




# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', default="./sub_abdomenMRCT_list.txt",help='line-seperated list of training files')
parser.add_argument('--seg-atlas', default="./imagesTr/AbdomenMRCT_0001_0001.nii.gz",
                    help='seg-atlas filename (default: data/atlas_norm.npz)')


parser.add_argument('--val-t1-list', default="./sub_abdomenMRCT_val_list.txt",help='line-seperated list of val files')
parser.add_argument('--val-seg-list', default="./sub_abdomenMRCT_val_seg_list.txt",help='line-seperated list of val files')

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
parser.add_argument('--gpu', default='4', help='GPU ID number(s), comma-separated (default: 0)')
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

# # recorder
# recorder = make_recorder("Log/", args.lr, args.weight, args.image_loss)

# 加载训练数据
train_files = HMCNet.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
train_seg_files = HMCNet.py.utils.read_file_list(args.train_seg_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = HMCNet.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    seg_atlas = HMCNet.py.utils.load_volfile(args.seg_atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = HMCNet.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
    train_generator = HMCNet.generators.seg_scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
    train_seg_generator = HMCNet.generators.seg_scan_to_atlas(train_seg_files, seg_atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = HMCNet.generators.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    seg_generator = HMCNet.generators.semisupervised(
    train_files,train_seg_files,batch_size=args.batch_size)

# # 加载验证集的数据

val_t1_files = HMCNet.py.utils.read_file_list(args.val_t1_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
val_seg_files = HMCNet.py.utils.read_file_list(args.val_seg_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)                                       
assert len(val_t1_files) > 0, 'Could not find any val data.'

atlas,atlas_affine = HMCNet.py.utils.load_volfile("/home/boys/Datasets/OASIS_OAS1_0176_MR1/t1.nii.gz", np_var='vol',
                                  add_batch_axis=True, add_feat_axis=add_feat_axis,ret_affine=True)
seg_atlas = HMCNet.py.utils.load_volfile(args.seg_atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
val_t1_generator = HMCNet.generators.seg_scan_to_atlas(val_t1_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
val_seg_generator = HMCNet.generators.seg_scan_to_atlas(val_seg_files, seg_atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

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
    image_loss_func = HMCNet.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = HMCNet.losses.MSE().loss
elif args.image_loss == 'MI':
    image_loss_func = HMCNet.losses.MutualInformation()
elif args.image_loss == 'MIND':
    image_loss_func = HMCNet.losses.MINDLoss().loss
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
losses += [HMCNet.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

loss_mse = [HMCNet.losses.MSE().loss]

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


def train_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221(config=config):
    # 设置固定的随机种子
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="abdomenCT", config=config)
    print("Training CRNet 3layer")
    

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    

    
    # 准备model_layer4
    newshape = (192,160,192)    
    model_layer2 = HMCNet.networks.dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221(inshape=(192,160,192))
    # 传递mednext的参数
    # model_dict = model_layer2.state_dict()
    # checkpoint = torch.load("/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/best_model_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_359.pth")
    # state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model_layer2.load_state_dict(model_dict)

    model_layer2.to(device1)
    model_layer2.train()

    
    # swanlab.watch(model_layer2,log="all", log_freq=10)

    # # 冻结相应层
    # for n,param in model_layer4.named_parameters():
    #     if "enc_block_3" in n or "dec_block_3" in n or "down_3" in n or "up_3" in n or "bottleneck" in n or "enc_block_2" in n or "dec_block_2" in n or "down_2" in n or "up_2" in n or "enc_block_1" in n or "dec_block_1" in n or "down_1" in n or "up_1" in n:
    #         param.requires_grad = False

    # set optimizer
    optimizer = torch.optim.Adam(model_layer2.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 50
    best_metric = -1

    # for epoch in range(args.layer4_epochs):
    for epoch in range(600):

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        epoch_total_diceloss = []


        for step in range(100):


            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]
    
            flow_1,flow_2,flow_3,y_flow = model_layer2(inputs[0],inputs[1])
            
            

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT = HMCNet.torch.layers.SpatialTransformer(newshape,mode="bilinear").to(device1)
            y_pre = SPT(inputs[0], y_flow)
            y_pred = []
            y_pred.append(y_pre)
            y_pred.append(y_flow)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                if n==2:
                    '''对应的是雅克比行列式的情况, 自己添加的'''
                    pred_flow = y_pred[1]  # y_pred的最后一个是预测的变形场flow, shape=[B,C,W,H,D]
                    pred_flow = pred_flow.permute(0, 3, 2, 4, 1)  # 为了适应雅克比行列式的输入shape=[B,H,W,D,C]
                    curr_loss = loss_function(pred_flow)
                else:
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            # loss_list.append(curr_loss.item())
            # loss += curr_loss


            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]}")


            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch':epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "best_model_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_" + str(epoch) + ".pth"
            Dice = 0
            mean_dice = 0
            num = 0
            model_layer2.eval()
            print("---------验证集的Dice----------")
            SPT = HMCNet.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                        seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # _,layer1_flow,_,_ = model_layer1(inputs_moving[0], inputs_fixed[0])
                            flow_1,flow_2,flow_3,flow = model_layer2(inputs_moving[0],inputs_fixed[0])
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice,dice_region = HMCNet.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0],dim=5)
                            if dice < 1:
                                print("dice", dice)
                                Dice += dice
                                # print(Dice)
                                num += 1
                print("----------换fixed image---------")
        
            metric = Dice / num
            swanlab.log({"mean_dice": metric, 'epoch':epoch})

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                torch.save(model_layer2.state_dict(),os.path.join(model_dir, name))
                # torch.save(model_layer2.state_dict(),os.path.join(model_dir, 'best_model_CRNet_3layer.pth'))
                # torch.save(model,os.path.join(model_dir, 'best_model_MedNext.pt'))
                print("saved new best metric MedNext layer4")

            print(f"Current epoch: {epoch+1} current dice in val_data: {metric} ")
            print(f"Best dice: {best_metric} at epoch {best_metric_epoch}")
            swanlab.log({"best_mean_dice": best_metric, 'epoch':epoch})
        torch.cuda.empty_cache()
    print("layer4 训练完成！")


def train_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_diff(config=config):
    # 设置固定的随机种子
    set_random_seed(42)  # 你可以将42替换为任何其他整数种子值

    run = swanlab.init(project="abdomenCT", config=config)
    print("Training CRNet 3layer")
    

    '''注意检查是否需要多卡模型并行训练'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device1 = torch.device('cuda:0')  # 分别表示用os.environ里面的第一个和第二个编号的gpu
    

    
    # 准备model_layer4
    newshape = (192,160,192)    
    model_layer2 = HMCNet.networks.dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_diff(inshape=(192,160,192))
    # 传递mednext的参数
    # model_dict = model_layer2.state_dict()
    # checkpoint = torch.load("/home/boys/project/voxelmorph/voxelmorph_code/models/comparison_methods/abdomenCT/best_model_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_359.pth")
    # state_dict = {k:v for k,v in checkpoint.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model_layer2.load_state_dict(model_dict)

    model_layer2.to(device1)
    model_layer2.train()

    
    # swanlab.watch(model_layer2,log="all", log_freq=10)

    # # 冻结相应层
    # for n,param in model_layer4.named_parameters():
    #     if "enc_block_3" in n or "dec_block_3" in n or "down_3" in n or "up_3" in n or "bottleneck" in n or "enc_block_2" in n or "dec_block_2" in n or "down_2" in n or "up_2" in n or "enc_block_1" in n or "dec_block_1" in n or "down_1" in n or "up_1" in n:
    #         param.requires_grad = False

    # set optimizer
    optimizer = torch.optim.Adam(model_layer2.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model_layer4.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # 验证集
    val_interval = 1
    best_metric = -1

    # for epoch in range(args.layer4_epochs):
    for epoch in range(600):

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        epoch_total_diceloss = []


        for step in range(1):


            step_start_time = time.time()
            # recorder.step += 1

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(seg_generator)
            inputs = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in y_true]
    
            flow_1,flow_2,flow_3,y_flow = model_layer2(inputs[0],inputs[1])
            
            

            # 对moving图像进行变换得到moved image
            # 定义空间转换层
            SPT = HMCNet.torch.layers.SpatialTransformer(newshape,mode="bilinear").to(device1)
            y_pre = SPT(inputs[0], y_flow)
            y_pred = []
            y_pred.append(y_pre)
            y_pred.append(y_flow)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                if n==2:
                    '''对应的是雅克比行列式的情况, 自己添加的'''
                    pred_flow = y_pred[1]  # y_pred的最后一个是预测的变形场flow, shape=[B,C,W,H,D]
                    pred_flow = pred_flow.permute(0, 3, 2, 4, 1)  # 为了适应雅克比行列式的输入shape=[B,H,W,D,C]
                    curr_loss = loss_function(pred_flow)
                else:
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    loss_list.append(curr_loss.item())
                    loss += curr_loss

            # loss_list.append(curr_loss.item())
            # loss += curr_loss


            print(f"loss: {loss_list[0] + loss_list[1]:.4f}  sim_loss: {loss_list[0]:.4f}  reg_loss: {loss_list[1]}")


            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            # get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        # scheduler.step()
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        swanlab.log({"total_loss": np.mean(epoch_total_loss), 'epoch':epoch})

        # # 对验证集进行测试

        if (epoch + 1) % val_interval == 0:
            name = "best_model_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_diff_" + str(epoch) + ".pth"
            Dice = 0
            mean_dice = 0
            num = 0
            model_layer2.eval()
            print("---------验证集的Dice----------")
            SPT = HMCNet.torch.layers.SpatialTransformer(inshape, mode="nearest").to(device1)
            for step in range(len(val_t1_files)):
                # 提取t1数据
                inputs_fixed, y_true = next(val_t1_generator)
                inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_fixed]
                # 提取对应t1的seg数据
                seg_inputs_fixed, seg_y_true = next(val_seg_generator)
                seg_inputs_fixed = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in seg_inputs_fixed]
                for step in range(len(val_t1_files)):
                    # 提取t1数据
                    inputs_moving, y_true = next(val_t1_generator)
                    inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in inputs_moving]
                    # 提取对应t1的seg数据
                    seg_inputs_moving, seg_y_true = next(val_seg_generator)
                    seg_inputs_moving = [torch.from_numpy(d).to(device1).float().permute(0, 4, 1, 2, 3) for d in
                                        seg_inputs_moving]
                    if step < len(val_t1_files) - 1:
                        with torch.no_grad():
                            # _,layer1_flow,_,_ = model_layer1(inputs_moving[0], inputs_fixed[0])
                            flow_1,flow_2,flow_3,flow = model_layer2(inputs_moving[0],inputs_fixed[0])
                            seg_pre = SPT(seg_inputs_moving[0], flow)
                            dice,dice_region = HMCNet.torch.losses.compute_dice(seg_pre, seg_inputs_fixed[0],dim=5)
                            if dice < 1:
                                print("dice", dice)
                                Dice += dice
                                # print(Dice)
                                num += 1
                print("----------换fixed image---------")
        
            metric = Dice / num
            swanlab.log({"mean_dice": metric, 'epoch':epoch})

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1

                torch.save(model_layer2.state_dict(),os.path.join(model_dir, name))
                # torch.save(model_layer2.state_dict(),os.path.join(model_dir, 'best_model_CRNet_3layer.pth'))
                # torch.save(model,os.path.join(model_dir, 'best_model_MedNext.pt'))
                print("saved new best metric MedNext layer4")

            print(f"Current epoch: {epoch+1} current dice in val_data: {metric} ")
            print(f"Best dice: {best_metric} at epoch {best_metric_epoch}")
            swanlab.log({"best_mean_dice": best_metric, 'epoch':epoch})
        torch.cuda.empty_cache()
    print("layer4 训练完成！")







if __name__ == "__main__":
    # train_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221() #  2
    train_dual_pyramid_VxmDense_tripleTrans_AMFFM_iterative2221_diff() # 3




    





    