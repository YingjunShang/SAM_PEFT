import os
import cv2
import numpy as np

def npy_to_jpg(npy_file, output_folder, normalize=True):
    """
    将 .npy 文件中的数据保存为 .jpg 文件。
    :param npy_file: 输入的 .npy 文件路径。
    :param output_folder: 输出 .jpg 文件的文件夹路径。
    :param normalize: 是否对数据进行归一化到 0-255。
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载 .npy 文件
    data = np.load(npy_file)

    # 判断数据维度
    if len(data.shape) == 2:  # 灰度图
        images = [data]
    elif len(data.shape) == 3:
        if data.shape[-1] in [1, 3]:  # 单帧（单通道或三通道）
            images = [data]
        else:  # 多帧
            images = [data[i] for i in range(data.shape[0])]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    # 保存每帧为 .jpg
    for idx, img in enumerate(images):
        if len(img.shape) == 2:  # 灰度图像
            img = img.astype(np.float32)
            if normalize:
                img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
        elif len(img.shape) == 3 and img.shape[-1] == 1:  # 单通道扩展为灰度
            img = img[:, :, 0]
        elif len(img.shape) == 3 and img.shape[-1] == 3:  # 彩色图像
            if normalize:
                img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # 保存为 .jpg 文件
        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(npy_file))[0]}_{idx:03d}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    # 示例：转换 npy 文件
    npy_file_path = "CT_Abd_FLARE22_Tr_0019-023.npy"  # 输入的 .npy 文件路径
    output_folder_path = "jpg"  # 输出 .jpg 文件夹
    npy_to_jpg(npy_file_path, output_folder_path)
