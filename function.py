
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import monai

import cfg
#import models.sam.utils.transforms as samtrans
import pytorch_ssim
#from models.discriminatorlayer import discriminator
from utils import *

# from lucent.modelzoo.util import get_model_layers
# from lucent.optvis import render, param, transform, objectives
# from lucent.modelzoo import inceptionv1

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
seed = torch.randint(1,11,(args.b,7))

torch.backends.cudnn.benchmark = True
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
scaler = torch.cuda.amp.GradScaler()
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
# cross entropy loss
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks
    
    
def train_sam(args, net: nn.Module, optimizer, train_loader,
          epoch, writer, schedulers=None, vis = 50):
    losses = []
    hard = 0
    epoch_loss = 0
    iter_num = 0
    # train mode
    net.train()
    optimizer.zero_grad()
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    if args.thd:
        lossfunc = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    else:
        lossfunc = criterion_G
    # 训练开始，train_loader是经过Dataloader()之后的nice_train_loader
    for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_loader)):
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        #print("boxes, ", boxes_np.shape) # (32, 4)
        #print("image, ", image.size()) # torch.Size([32, 3, 1024, 1024])
        #print("gt2D, ", gt2D.size()) # torch.Size([32, 1, 1024, 1024])
        
        total_params = 0  # 总参数量
        trainable_params = 0  # 可训练参数量
        
        for n, value in net.image_encoder.named_parameters(): 
            if "Adapter" not in n:
                value.requires_grad = False  # 冻结该参数
            else:
                value.requires_grad = True
                trainable_params += value.numel()  # 统计可训练的参数量
                
            total_params += value.numel()  # 统计总参数量
        
        for n, value in net.image_encoder.named_parameters(): 
            if "Adapter" not in n:
                value.requires_grad = False  # 冻结该参数
            else:
                value.requires_grad = True
                trainable_params += value.numel()  # 统计可训练的参数量
                
            total_params += value.numel()  # 统计总参数量

        #print(f"Total parameters: {total_params}")  # 输出总参数量100322304
        #print(f"Trainable parameters: {trainable_params}")  # 输出可训练参数量10651392 (1/9.5)
        #print(f"Frozen parameters: {total_params - trainable_params}")  # 输出冻结的参数量89670912


        
        ## AMP
        with torch.autocast(device_type="cuda", dtype=torch.float16): # 下面是 MedSAMtrain.py里的class 的 forward
            image_embedding = net.image_encoder(image)  # (B, 256, 64, 64)
            with torch.no_grad():
                box_torch = torch.as_tensor(boxes_np, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = net.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
                )
            low_res_masks, _ = net.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=net.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False, # med-sam-adapter -> (args.multimask_output > 1),
            )
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

            medsam_pred = ori_res_masks # medsam_model(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                medsam_pred, gt2D.float()
            )
            #loss = lossfunc(pred, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        iter_num += 1
        
        epoch_loss1 = epoch_loss / (step+1)
        losses.append(epoch_loss1)
        
        '''
        if vis:
            if iter_num % vis == 0:
                namecat = 'Train'
                for na in name[:2]:
                    namecat = namecat + na.split('/')[-1].split('.')[0] + '+'
                vis_image(image,ori_res_masks,gt2D, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=showp)
        '''
    return loss

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,)*args.multimask_output*2
    rater_res = [(0,0,0,0) for _ in range(6)]
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    device = GPUdevice

    for step, (image, gt2D, boxes, _) in enumerate(tqdm(val_loader)):
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        #print("boxes, ", boxes_np.shape) # (32, 4)
        #print("image, ", image.size()) # torch.Size([32, 3, 1024, 1024])
        #print("gt2D, ", gt2D.size()) # torch.Size([32, 1, 1024, 1024])
        
        total_params = 0  # 总参数量
        trainable_params = 0  # 可训练参数量
        
        for n, value in net.image_encoder.named_parameters(): 
            if "Adapter" not in n:
                value.requires_grad = False  # 冻结该参数
            else:
                value.requires_grad = True
                trainable_params += value.numel()  # 统计可训练的参数量
                
            total_params += value.numel()  # 统计总参数量
        
        for n, value in net.image_encoder.named_parameters(): 
            if "Adapter" not in n:
                value.requires_grad = False  # 冻结该参数
            else:
                value.requires_grad = True
                trainable_params += value.numel()  # 统计可训练的参数量
                
            total_params += value.numel()  # 统计总参数量

        #print(f"Total parameters: {total_params}")  # 输出总参数量100322304
        #print(f"Trainable parameters: {trainable_params}")  # 输出可训练参数量10651392 (1/9.5)
        #print(f"Frozen parameters: {total_params - trainable_params}")  # 输出冻结的参数量89670912


        
        ## AMP
        with torch.autocast(device_type="cuda", dtype=torch.float16): # 下面是 MedSAMtrain.py里的class 的 forward
            image_embedding = net.image_encoder(image)  # (B, 256, 64, 64)
            with torch.no_grad():
                box_torch = torch.as_tensor(boxes_np, dtype=torch.float32, device=image.device)
                if len(box_torch.shape) == 2:
                    box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = net.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
                )
            low_res_masks, _ = net.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=net.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False, # med-sam-adapter -> (args.multimask_output > 1),
            )
            ori_res_masks = F.interpolate(
                low_res_masks,
                size=(image.shape[2], image.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

            medsam_pred = ori_res_masks # medsam_model(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                medsam_pred, gt2D.float()
            )
            
            temp = eval_seg(medsam_pred, gt2D, threshold)
            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])


    if args.evl_chunk:
        n_val = n_val * (imgsw.size(-1) // evl_ch)

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def transform_prompt(coord,label,h,w):
    coord = coord.transpose(0,1)
    label = label.transpose(0,1)

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    num_pts = coord.shape[2]
    rescaled_batched_points = get_rescaled_pts(coord, h, w)

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points,label


def get_rescaled_pts(batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * 1024 / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * 1024 / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )