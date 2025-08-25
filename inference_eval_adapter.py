import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

from os.path import join
from skimage import transform
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#from dataset import *
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import random
import glob

import cfg
import function
#from models.discriminator import discriminator
#from dataset import *
from utils import *
join = os.path.join
import cv2


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join("/home/xietong/SSN/med-sam-adapter2", args.exp_name + "-" + run_id)



def main():
    args = cfg.parse_args()
    test_dataset = NpyDataset("/home/xietong/SSN/med-sam-adapter2/data/npy_val/CT_Abd")
    print("Number of testing samples: ", len(test_dataset))
    # net = SAM
    GPUdevice = torch.device('cuda', args.gpu_device)
    args.weights = "/home/xietong/SSN/med-sam-adapter2/logs/med-sam-adapter2_2024_12_12_23_23_58/Model/best_checkpoint.pth"
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    weights = torch.load(args.weights)
    net.load_state_dict(weights,strict=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    '''load pretrained model 恢复训练Fintune'''  
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {epoch})')
    net.eval()
    args.path_helper = set_log_dir('/home/xietong/SSN/med-sam-adapter2/logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
    args.b = 1
    '''
    nice_test_loader = DataLoader(
        test_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    '''

    '''checkpoint path and tensorboard'''
    TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")
    LOG_DIR = '/home/xietong/SSN/med-sam-adapter2/runs'
    #use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            LOG_DIR, args.net, TIME_NOW))

    '''
    net.eval()
    iou_score, dice_score = function.validation_sam(args, nice_test_loader, epoch, net, writer)
    logger.info(f'IOU: {iou_score}, DICE: {dice_score} || @ epoch {epoch}, Time: {datetime.now().strftime("%Y%m%d-%H%M")}.')
    '''
    ##############################################################
    
    img_path = "/home/xietong/SSN/med-sam-adapter2/data/npy_val/CT_Abd"
    seg_path = "/home/xietong/SSN/med-sam-adapter2/data/comparison_SAM_seg_output_val"
    gt_path_files = sorted(glob.glob(join(img_path, '**/*.npz'), recursive=True))
    print('find {} files'.format(len(gt_path_files)))
    image_size = 1024
    bbox_shift = 20

    # predict npz files and save results
    for gt_path_file in tqdm(gt_path_files):
        npz_name = os.path.basename(gt_path_file)
        task_folder = gt_path_file.split('/')[-2]
        os.makedirs(join(seg_path, task_folder), exist_ok=True)
        npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
        img_3D = npz_data['imgs'] # (Num, H, W) (98, 512, 512) uint8
        #print("img_3D, ", img_3D.dtype)
        gt_3D = npz_data['gts'] # (Num, 256, 256) (98, 512, 512)
        spacing = npz_data['spacing']
        seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num, 256, 256)
        #print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww", gt_3D[0].size())
        for i in range(img_3D.shape[0]):
            img_2d = img_3D[i,:,:] # (H, W, 3)
            img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1)
            
            resize_img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC) # (1024, 1024, 3) uint8
            
            resize_img_1024_tensor = torch.from_numpy(resize_img_1024) 

            # 改变维度顺序，从 (H, W, C) 到 (C, H, W)
            resize_img_1024_tensor = resize_img_1024_tensor.permute(2, 0, 1)

            # 在第一个维度上增加一个维度，得到形状为 [1, C, H, W]
            resize_img_1024_tensor = resize_img_1024_tensor.unsqueeze(0)
            resize_img_1024_tensor = resize_img_1024_tensor.to(dtype=torch.float16)
            resize_img_1024_tensor = resize_img_1024_tensor.to(GPUdevice)
            #print("resize_img_1024, ", resize_img_1024.shape)
            #predictor.set_image(resize_img_1024.astype(np.uint8)) # conpute the image embedding only once
            #print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq     ", resize_img_1024.size)
            gt = gt_3D[i,:,:] # (H, W)
            gt_1024 = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST) # (1024, 1024)
            label_ids = np.unique(gt)[1:]
            for label_id in label_ids:
                gt_1024_label_id = np.uint8(gt_1024 == label_id) # only one label, (256, 256)
                y_indices, x_indices = np.where(gt_1024_label_id > 0)
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                # add perturbation to bounding box coordinates
                H, W = gt_1024_label_id.shape
                x_min = max(0, x_min - bbox_shift)
                x_max = min(W, x_max + bbox_shift)
                y_min = max(0, y_min - bbox_shift)
                y_max = min(H, y_max + bbox_shift)
                bboxes1024 = np.array([x_min, y_min, x_max, y_max])
                boxes_np=bboxes1024[None, :] # (1,4)
                #sam_mask, _, _ = predictor.predict(point_coords=None, point_labels=None, box=bboxes1024[None, :], multimask_output=False) #1024x1024, bool
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
                # resize_img_1024 相当于要输入进med-sam-adapter2网络的img
                with torch.autocast(device_type="cuda", dtype=torch.float16): # 下面是 MedSAMtrain.py里的class 的 forward
                    image_embedding = net.image_encoder(resize_img_1024_tensor)  # 应该是 torch.Size([32, 3, 1024, 1024])
                    with torch.no_grad():
                        box_torch = torch.as_tensor(boxes_np, dtype=torch.float32, device=resize_img_1024_tensor.device)
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
                        size=(resize_img_1024_tensor.shape[2], resize_img_1024_tensor.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )

                    medsam_pred = ori_res_masks # torch.Size([1, 1, 1024, 1024])
                
                #print("hhh ", gt) # (98, 512, 512) uint8
                medsam_predxxx = medsam_pred.squeeze(0).squeeze(0)
                thresholded_pred = (medsam_predxxx > 4).float()
                thresholded_pred_image = (thresholded_pred * 255).cpu().byte()
                image = Image.fromarray(thresholded_pred_image.numpy())
                image.save('thresholded_image.png')
                
                
                min_value = torch.min(medsam_pred)
                max_value = torch.max(medsam_pred)
                print(f"The range of the array is from {min_value} to {max_value}")
                
                
                
                medsam_pred_np = medsam_pred[0].detach().cpu().numpy().astype(np.uint8) # (1, 1024, 1024)
                '''
                medsam_pred_resized = transform.resize(medsam_pred_np, 
                                                       (gt.shape[-2], gt.shape[-1]), 
                                                       order=0, 
                                                       preserve_range=True, 
                                                       mode='constant', 
                                                       anti_aliasing=False) # (512, 512, 1024)
                '''
                medsam_pred_512_512 = cv2.resize(medsam_pred_np[0], (512, 512), interpolation=cv2.INTER_AREA)
                #np.savetxt('aaa.txt', medsam_pred_512_512, fmt='%f')
                #unique_values, counts = np.unique(medsam_pred_512_512, return_counts=True)
                #print("Unique values:", unique_values)
                #print("Counts of unique values:", counts)
                
                #a = [127,128,129]
                #xxximg = np.isin(medsam_pred_512_512, a).astype(np.uint8)
                #xxximg = 255 * xxximg
                #plt.imshow(xxximg, cmap='gray')
                #plt.axis('off')  # 不显示坐标轴
                #plt.show()
                #plt.imsave('output_image127.png', xxximg, cmap='gray')
                
                
                seg_3D[i, medsam_pred_512_512>0] = label_id
                #medsam_pred = transform.resize(medsam_pred[0].astype(np.uint8), (gt.shape[-2], gt.shape[-1]), order=0, preserve_range=True, mode='constant', anti_aliasing=False) # (256, 256)
                #seg_3D[i, medsam_pred>0] = label_id
                #logger.info(f'IOU: {iou_score}, DICE: {dice_score} || @ epoch {epoch}, Time: {datetime.now().strftime("%Y%m%d-%H%M")}.')
        
        np.savez_compressed(join(seg_path, task_folder, npz_name), segs=seg_3D, gts=gt_3D, spacing=spacing) # save spacing for metric computation
        

if __name__ == '__main__':
    main()
#211635 背景
#50509 器官 191 算器官
#+=262144=512*512