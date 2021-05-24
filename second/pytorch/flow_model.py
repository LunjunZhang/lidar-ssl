
import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle

from flow.models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from flow.models import multiScaleLoss
from pathlib import Path
from collections import defaultdict

import flow.transforms as transforms
import flow.datasets as datasets
import flow.cmd_args as cmd_args

# from flow.main_utils import *
from flow.utils import geometry
from flow.evaluation_utils import evaluate_2d, evaluate_3d

'''
#import ipdb; ipdb.set_trace()
if 'NUMBA_DISABLE_JIT' in os.environ:
    del os.environ['NUMBA_DISABLE_JIT']
'''

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    vgrid = vgrid.float()
    # import pdb; pdb.set_trace()
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask, mask

global args 
args = cmd_args.parse_args_from_yaml('./flow/config_evaluate.yaml')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1,2,3'


model = PointConvSceneFlow()

#load pretrained model
pretrain = args.ckpt_dir + args.pretrain
model.load_state_dict(torch.load(pretrain))

print('load model %s'%pretrain)

model.cuda()
