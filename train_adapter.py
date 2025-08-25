import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from datetime import datetime

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
    train_dataset = NpyDataset("/home/xietong/SSN/med-sam-adapter2/data/npy/CT_Abd")
    print("Number of training samples: ", len(train_dataset))
    test_dataset = NpyDataset("/home/xietong/SSN/med-sam-adapter2/data/npy_val/CT_Abd")
    print("Number of testing samples: ", len(test_dataset))
    # net = SAM
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution = args.distributed)
    if args.pretrain: # Finetune#####################################################
        weights = torch.load(args.pretrain)
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
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'],strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')

    args.path_helper = set_log_dir('/home/xietong/SSN/med-sam-adapter2/logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    nice_train_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    nice_test_loader = DataLoader(
        test_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )


    '''checkpoint path and tensorboard'''
    TIME_NOW = datetime.now().strftime("%F_%H-%M-%S.%f")
    LOG_DIR = '/home/xietong/SSN/med-sam-adapter2/runs'
    #use tensorboard
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            LOG_DIR, args.net, TIME_NOW))


    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_iou = 0.0
    best_dice = 0.0
    best_loss = 1e10
    losses = []
    EPOCH = 100 #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    for epoch in range(EPOCH+1):
        '''
        if epoch and epoch < 5:
            tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}, Time: {datetime.now().strftime("%Y%m%d-%H%M")}.')
        '''
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch, writer)
        writer.add_scalar("loss",loss,epoch)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}, Time: {datetime.now().strftime("%Y%m%d-%H%M")}.')
        time_end = time.time()
        print('time_for_training ', (time_end - time_start)/60, " mins")
        
        net.eval()
        tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, epoch, net, writer)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}|| @ epoch {epoch}, Time: {datetime.now().strftime("%Y%m%d-%H%M")}.')
        
        '''
        
        is_best = False
        filename = f"checkpoint_epoch_{epoch}.pth"
        if (epoch > 10 and epoch % 10 == 0):
                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename=filename)
        if loss < best_loss:
            is_best = True
            best_loss = loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_tol': best_dice,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename="best_checkpoint.pth")
        '''
        scheduler.step() # 更新学习率
        #losses.append(loss)

    writer.close()


if __name__ == '__main__':
    main()
