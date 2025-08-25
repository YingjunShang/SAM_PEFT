import os
import cv2
import numpy as np

def npz_to_jpg(npz_file, output_folder):
    """
    将 npz 文件中的图像保存为 jpg 文件。
    :param npz_file: 输入的 npz 文件路径。
    :param output_folder: 保存 jpg 文件的文件夹。
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载 npz 文件
    data = np.load(npz_file, allow_pickle=True)
    print(data.files)
    '''
    # 遍历所有数组
    for key in data.keys():
        array = data[key]
        if len(array.shape) == 2 or (len(array.shape) == 3 and array.shape[-1] in [1, 3]):
            # 处理每个图像帧
            if len(array.shape) == 2:  # 灰度图
                array = (array * 255).astype(np.uint8)  # 归一化到 0-255
            elif array.shape[-1] == 1:  # 单通道扩展为灰度图
                array = array[:, :, 0]
            elif array.shape[-1] == 3:  # 彩色图像
                array = array.astype(np.uint8)

            # 保存每帧为 jpg 文件
            file_name = f"{key}.jpg"
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, array)
            print(f"Saved: {output_path}")

        elif len(array.shape) == 3:  # 3D 图像数据
            for i in range(array.shape[0]):
                frame = array[i]
                if len(frame.shape) == 2:  # 灰度
                    frame = (frame * 255).astype(np.uint8)
                elif frame.shape[-1] == 1:  # 单通道扩展为灰度
                    frame = frame[:, :, 0]
                elif frame.shape[-1] == 3:  # 彩色图像
                    frame = frame.astype(np.uint8)
                
                # 保存每帧为 jpg 文件
                file_name = f"{key}_{i:03d}.jpg"
                output_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_path, frame)
                print(f"Saved: {output_path}")
                '''

if __name__ == "__main__":
    # 示例：转换 npz 文件
    npz_file_path = "D:\All_courses\CV\project\SAMnpz\CT_Abd_FLARE22_Tr_0001.npz"  # 输入的 npz 文件路径
    output_folder_path = "D:\All_courses\CV\project\SAM_jpg"  # 输出 jpg 文件夹
    npz_to_jpg(npz_file_path, output_folder_path)
# 026-029