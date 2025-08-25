import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from os import makedirs
join = os.path.join
import torch
from src.segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
from src.segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


@torch.no_grad()
def sam_adapter_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


#  load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="assets/img_demo.png",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=str,
    default='[95, 255, 190, 350]',
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
args = parser.parse_args()


img_path = "/home/xietong/SSN/med-sam-adapter2/data/npy_val/CT_Abd"
seg_path = "/home/xietong/SSN/med-sam-adapter2/data/comparison_SAM_seg_output_val"
makedirs(seg_path, exist_ok=True)
device = torch.device("cuda:0")
gt_path_files = sorted(glob.glob(join(img_path, '**/*.npz'), recursive=True))
print('find {} files'.format(len(gt_path_files)))
image_size = 1024
bbox_shift = 20
    
    
device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()
#predictor = SamPredictor(sam_model)



for gt_path_file in tqdm(gt_path_files):
    npz_name = os.path.basename(gt_path_file)
    task_folder = gt_path_file.split('/')[-2]
    os.makedirs(join(seg_path, task_folder), exist_ok=True)
    npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3D = npz_data['imgs'] # (Num, H, W) (98, 512, 512)
    gt_3D = npz_data['gts'] # (Num=98, 512, 512)
    spacing = npz_data['spacing']
    seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num=98, 512, 512)
    
   
    for i in range(img_3D.shape[0]):
        img_2d = img_3D[i,:,:] # (H, W, 3)
        img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1) # (512, 512, 3)
        
        #resize_img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC) # (1024, 1024, 3)
        #predictor.set_image(resize_img_1024.astype(np.uint8)) # conpute the image embedding only once
        resize_img_1024 = transform.resize(
                    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
                    ).astype(np.uint8)
        resize_img_1024 = (resize_img_1024 - resize_img_1024.min()) / np.clip(
                    resize_img_1024.max() - resize_img_1024.min(), a_min=1e-8, a_max=None
                    )
        img_1024_tensor = (
                    torch.tensor(resize_img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    )
        gt = gt_3D[i,:,:] # (H, W)
        gt_1024 = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST) # (1024, 1024)
        label_ids = np.unique(gt)[1:]
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
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
            bboxes1024 = np.expand_dims(bboxes1024, axis=0)
            #print(image_embedding.shape) # ([1, 256, 64, 64])
            medsam_seg = sam_adapter_inference(medsam_model, image_embedding, bboxes1024, H, W) # Unique values: [0 1] (1024, 1024)
            #sam_mask, _, _ = predictor.predict(point_coords=None, point_labels=None, box=bboxes1024[None, :], multimask_output=False) #1024x1024, bool
            medsam_seg = transform.resize(medsam_seg[0].astype(np.uint8), (gt.shape[-2], gt.shape[-1]), order=0, preserve_range=True, mode='constant', anti_aliasing=False) # (256, 256)
            seg_3D[i, medsam_seg>0] = label_id
    np.savez_compressed(join(seg_path, task_folder, npz_name), segs=seg_3D, gts=gt_3D, spacing=spacing) # save spacing for metric computation
    

# @@@@@@@@@@@@@@@@@@@@@@@@
img_np = io.imread(args.data_path) # (512, 512)

if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) # (512, 512, 3)
else:
    img_3c = img_np
H, W, _ = img_3c.shape
#  image preprocessing
img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8) # (1024, 1024, 3)





img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3) (1024, 1024, 3) 

img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
) # torch.Size([1, 3, 1024, 1024])

box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]]) # (1, 4)

# transfer box_np t0 1024x1024 scale
box_1024 = box_np / np.array([W, H, W, H]) * 1024  # (1, 4)
#print(box_1024)
with torch.no_grad():
    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
print(image_embedding.shape)
medsam_seg = sam_adapter_inference(medsam_model, image_embedding, box_1024, H, W) # Unique values: [0 1]
unique_values, counts = np.unique(medsam_seg, return_counts=True)

io.imsave(
    join(args.seg_path, "seg_" + os.path.basename(args.data_path)),
    medsam_seg,
    check_contrast=False,
)
'''
# 直接打印的话图片是黑的，要执行：
img_path = 'assets/seg_img_demo.png'
img =plt.imread(img_path)
 
plt.imshow(img)
plt.show()
plt.savefig('output_image.png', dpi=300, bbox_inches='tight')
'''