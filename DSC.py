import numpy as np
import os

# 定义文件路径范围
file_paths = [f"D:/All_courses/CV/project/SAMnpz/CT_Abd_FLARE22_Tr_{i:04d}.npz" for i in range(1, 41)]

# 类别列表
FLARE_LABELS_LIST = ['liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 
                     'inferior vena cava', 'right adrenal gland', 'left adrenal gland', 
                     'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney']

# 初始化结果存储
results = []

def compute_dsc(gt, pred): # 512 
    """计算Dice Similarity Coefficient (DSC)"""
    intersection = np.sum((gt > 0) & (pred > 0))  # 交集
    gt_sum = np.sum(gt > 0)  # 真实掩码区域
    pred_sum = np.sum(pred > 0)  # 预测掩码区域
    if (gt_sum + pred_sum) > 0:
        return 2.0 * intersection / (gt_sum + pred_sum)
    else:
        #print("出现了该死的0", gt_sum, ' ', pred_sum)
        return 0.0

# 遍历所有类别（去除背景 0 类别）
for label_id, label_name in enumerate(FLARE_LABELS_LIST, start=1):
    dsc_per_image = []  # 存储每张图片的DSC
    # 遍历所有文件
    for file_path in file_paths:
        # 加载数据
        data = np.load(file_path)
        segs = data['segs']  # 预测掩码 (101, 512, 512)
        gts = data['gts']    # 真实掩码 (101, 512, 512)
        
        # 遍历每张图像
        for i in range(segs.shape[0]):  # 第i张图像
            # 获取每张图像的标签掩码
            gt_label_mask = (gts[i] == label_id).astype(np.uint8) # T, F
            pred_label_mask = (segs[i] == label_id).astype(np.uint8)
            
            # 计算每张图像的DSC
            dsc = compute_dsc(gt_label_mask, pred_label_mask)
            if dsc != 0:
                dsc_per_image.append(dsc)

    # 计算所有非零元素的均值
    print(len(dsc_per_image))
    print()
    avg_dsc = np.mean(dsc_per_image)
    

    # 保存结果
    results.append((label_name, avg_dsc))

# 输出结果到文件
with open("SAMdsc_output.txt", "w") as file:
    file.write("Average DSC for each organ across all files:\n")
    for label_name, avg_dsc in results:
        file.write(f"{label_name}: Mean DSC = {avg_dsc:.4f}\n")

print("Results have been written to 'SAMdsc_output.txt'.")
