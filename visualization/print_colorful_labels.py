import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 加载数据
data = np.load("D:\All_courses\CV\project\MedSAMnpz\CT_Abd_FLARE22_Tr_0026.npz")
segs = data['segs']  # 预测的分割结果
gts = data['gts']    # 实际的标签

# 13个类别的标签（包括背景）
labels = ['background', 'liver', 'right kidney', 'spleen', 'pancreas', 'aorta', 'inferior vena cava',
          'right adrenal gland', 'left adrenal gland', 'gallbladder', 'esophagus', 'stomach', 'duodenum', 'left kidney']

# 创建一个颜色映射：13个类别的颜色
colors = [
    (0, 0, 0),         # background: black
    (1, 0, 0),         # liver: red √
    (0, 1, 0),         # right kidney: green √
    (0, 0, 1),         # spleen: blue √
    (1, 1, 0),         # pancreas: yellow
    (1, 0, 1),         # aorta: magenta 品红 √
    (0, 1, 1),         # inferior vena cava: cyan 青色
    (0.5, 0, 0),       # right adrenal gland: dark red √
    (0, 0.5, 0),       # left adrenal gland: dark green √
    (0, 0, 0.5),       # gallbladder: dark blue √
    (0.5, 0.5, 0),     # esophagus: olive 橄榄色 √
    (0.5, 0, 0.5),     # stomach: purple √
    (0, 0.5, 0.5),     # duodenum: teal 青绿色
    (0.7, 0.7, 0)      # left kidney: light yellow √
]

# 创建颜色映射器
cmap = ListedColormap(colors)

# 显示预测分割结果与真实标签
for i in range(segs.shape[0]):  # 遍历所有图片
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示预测结果
    ax1.imshow(segs[i], cmap=cmap)
    ax1.set_title(f"Prediction for image {i+1}")
    ax1.axis('off')  # 关闭坐标轴
    
    # 显示真实标签
    ax2.imshow(gts[i], cmap=cmap)
    ax2.set_title(f"Ground Truth for image {i+1}")
    ax2.axis('off')  # 关闭坐标轴
    
    plt.show()
