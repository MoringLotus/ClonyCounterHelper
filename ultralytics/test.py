import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 加载预训练模型（使用你自己的权重文件路径）
model = YOLO("/home/featurize/work/bhintern/ClonyCounterHelper/runs/detect/train2/weights/best.pt")  # 替换为你的权重文件路径

# 验证模型
results = model.val(
    data="/home/featurize/work/bhintern/ultralytics/ultralytics/cfg/datasets/AGAR.yaml",  # 数据集配置文件路径
    imgsz=1280,         # 图像大小
    batch=16,           # 批量大小
    device="cuda"       # 使用 GPU 设备（如果有）
)

# 获取 mAP 数据
all_ap = results.box.all_ap  # 获取所有 IoU 阈值下的 AP 值

# 检查 all_ap 是否为空
if all_ap is None or len(all_ap) == 0:
    raise ValueError("all_ap 数据为空，请检查模型验证过程是否正确！")

# IoU 阈值范围
ious = np.arange(0.5, 1.0, 0.05)  # 从 0.5 到 0.95，步长为 0.05

# 检查 all_ap 的形状是否正确
if all_ap.shape[1] != len(ious):
    raise ValueError(f"all_ap 的列数 {all_ap.shape[1]} 与 ious 的长度 {len(ious)} 不匹配，请检查数据！")

# 创建一个图形窗口
plt.figure(figsize=(10, 6))

# 遍历每个类别
for i in range(all_ap.shape[0]):
    # 获取当前类别的 AP 值
    ap = all_ap[i]
    
    # 计算当前类别的 mAP0.5-0.95
    map_05_95 = np.mean(ap)
    
    # 绘制当前类别的 mAP 曲线
    plt.plot(ious, ap, marker='o', linestyle='-', label=f'Class {i+1} (mAP0.5-0.95: {map_05_95:.4f})')

# 设置图表标题和标签
plt.xlabel('IoU Threshold', fontsize=14)
plt.ylabel('AP', fontsize=14)
plt.title('mAP0.5-0.95 Curves for All Classes', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.xticks(np.arange(0.5, 1.0, 0.05), fontsize=12)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
plt.tight_layout()

# 保存图表到本地
plt.savefig("/home/featurize/work/bhintern/ClonyCounterHelper/mAP0.5-0.95_curves.png")  # 替换为你希望保存的路径
plt.close()