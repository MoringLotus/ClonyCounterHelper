from ultralytics import YOLO
from collections import Counter
import os
import json

# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/runs/detect/train3/weights/best.pt")

# 定义图片和标签文件夹路径
image_folder = "/home/featurize/work/bhintern/yolodata/valid/images"
label_folder = "/home/featurize/work/bhintern/yolodata/valid/labels_ori"

# 获取文件夹中所有图片路径
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

# 类别 ID 与名称的映射
class_names = {
    0: "C.albicans",
    1: "B.subtilis",
    2: "S.aureus",
    3: "E.coli",
    4: "P.aeruginosa"
}

# 初始化统计字典
recall_stats = {class_name: {"TP": 0, "FN": 0} for class_name in class_names.values()}

# 遍历所有图片
for image_path in image_paths:
    # 获取图片文件名（不带扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 构造对应的 JSON 标签文件路径
    label_path = os.path.join(label_folder, f"{image_name}.json")
    
    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        print(f"Label file not found for image: {image_name}")
        continue
    
    # 读取 JSON 标签文件
    with open(label_path, 'r') as file:
        label_data = json.load(file)
    
    # 提取真实数量
    true_labels = [label["class"] for label in label_data.get("labels", [])]
    true_counts = Counter(true_labels)
    
    # 对单张图片进行推理
    result = model.predict(image_path)
    
    # 提取预测结果中的类别标签
    if result[0].boxes:
        class_labels = result[0].boxes.cls.cpu().numpy().astype(int)
        predicted_labels = [class_names[label_id] for label_id in class_labels]
        predicted_counts = Counter(predicted_labels)
    else:
        predicted_counts = Counter()
    
    # 更新统计信息
    for class_name in class_names.values():
        TP = min(true_counts.get(class_name, 0), predicted_counts.get(class_name, 0))  # 真正例
        FN = true_counts.get(class_name, 0) - TP  # 假反例
        recall_stats[class_name]["TP"] += TP
        recall_stats[class_name]["FN"] += FN

# 计算每个类别的召回率
print("Recall for each class:")
for class_name, stats in recall_stats.items():
    TP = stats["TP"]
    FN = stats["FN"]
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    print(f"Class: {class_name}, Recall: {recall:.4f} (TP: {TP}, FN: {FN})")