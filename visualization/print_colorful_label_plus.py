import numpy as np
import matplotlib.pyplot as plt
import cv2

# 标签和颜色映射
labels = [
    'background', 'liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava',
    'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney'
]

colors = [
    (0, 0, 0),         # background: black
    (1, 0, 0),         # liver: red
    (0, 1, 0),         # right kidney: green
    (0, 0, 1),         # spleen: blue
    (1, 1, 0),         # pancreas: yellow
    (1, 0, 1),         # aorta: magenta
    (0, 1, 1),         # inferior vena cava: cyan
    (0.5, 0, 0),       # right adrenal gland: dark red
    (0, 0.5, 0),       # left adrenal gland: dark green
    (0, 0, 0.5),       # gallbladder: dark blue
    (0.5, 0.5, 0),     # esophagus: olive
    (0.5, 0, 0.5),     # stomach: purple
    (0, 0.5, 0.5),     # duodenum: teal
    (0.7, 0.7, 0)      # left kidney: light yellow
]

# 读取图像和mask
def load_data(image_path, mask_path):
    img = np.load(image_path)  # img shape: (H, W, 3)
    mask = np.load(mask_path) # npz
    segs = mask['segs']    # 预测掩码 (101, 512, 512)
    gts = mask['gts']    # 真实掩码 (101, 512, 512)
    return img, segs[48], gts[48]

# 叠加颜色
def overlay_mask_on_image(img, mask, colors, alpha=0.5):
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    assert img.shape[:2] == mask.shape, "Image and mask dimensions do not match"
    overlay = img.copy()
    print(mask)
    for label_idx, color in enumerate(colors):
        if label_idx != 0:
            print("label_idx = ", label_idx)
            print("color = ", color)
            overlay[mask == label_idx] = (overlay[mask == label_idx] * (1 - alpha) + np.array(color) * 255 * alpha)

    # 转换为整数以显示图像
    overlay = overlay.astype(np.uint8)
    return overlay

# 可视化函数
def visualize(img, overlay_seg, overlay_gt):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(overlay_seg)
    axs[1].set_title("SAM2")
    axs[1].axis("off")

    axs[2].imshow(overlay_gt)
    axs[2].set_title("GT")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

def main(image_path, mask_path):
    img, seg, gt = load_data(image_path, mask_path)
    print("img.shape = ", img.shape) # (1024, 1024, 3)
    print("gt.shape = ", gt.shape) # (512, 512)
    resized_gt = cv2.resize(gt.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    resized_seg = cv2.resize(seg.astype(np.float32), (1024, 1024), interpolation=cv2.INTER_NEAREST)
    overlay_seg = overlay_mask_on_image(img, resized_seg, colors, alpha=0.5)
    overlay_gt = overlay_mask_on_image(img, resized_gt, colors, alpha=0.5)
    visualize(img, overlay_seg, overlay_gt)

image_path = "D:\All_courses\CV\project\imgs\CT_Abd_FLARE22_Tr_0001-048.npy"
mask_path = "D:\All_courses\CV\project\SAM2npz\CT_Abd_FLARE22_Tr_0001.npz"
main(image_path, mask_path)
