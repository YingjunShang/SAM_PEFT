import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from torchvision.utils import save_image
import src.utils as utils
from src.dataloader import DatasetSegmentation, collate_fn
from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

import argparse
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
import numpy as np
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
model_save_path = join("/home/xietong/SSN/med-sam-lora", args.exp_name + "-" + run_id)
rank = 512 # 2 4 6 8 16 32 64 128 256 512

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

    sam_lora = LoRA_sam(net, rank)  
    model = sam_lora.sam
    optimizer = Adam(model.image_encoder.parameters(), lr=1e-4, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    num_epochs = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set model to train and into the device
    model.train()
    model.to(device)

    total_loss = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for i, batch in enumerate(tqdm(nice_train_loader)):
          
          outputs = model(batched_input=batch,
                          multimask_output=False)

          stk_gt, stk_out = utils.stacking_batch(batch, outputs)
          stk_out = stk_out.squeeze(1)
          stk_gt = stk_gt.unsqueeze(1)
          loss = seg_loss(stk_out, stk_gt.float().to(device))
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          epoch_losses.append(loss.item())
        print(f'EPOCH: {epoch}')
    sam_lora.save_lora_parameters(f"lora_rank{rank}.safetensors")
