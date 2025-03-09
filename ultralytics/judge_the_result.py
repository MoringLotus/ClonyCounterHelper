import os
from ultralytics import YOLO

def read_label_file(label_path):
    """
    读取 YOLO 格式的标注文件，返回所有类别的数量统计。

    Args:
        label_path (str): 标注文件路径。

    Returns:
        dict: 每个类别的数量统计。
    """
    counts = {}
    if not os.path.exists(label_path):
        return counts  # 如果标注文件不存在，返回空字典

    with open(label_path, "r") as f:
        for line in f.readlines():
            class_idx, _, _, _, _ = map(float, line.strip().split())
            class_idx = int(class_idx)
            if class_idx in counts:
                counts[class_idx] += 1
            else:
                counts[class_idx] = 1
    return counts

def evaluate_model(image_folder, label_folder):
    """
    对文件夹中的所有图片进行推理，并计算每个类别的召回率。

    Args:
        image_folder (str): 图片文件夹路径。
        label_folder (str): 标注文件夹路径。

    Returns:
        dict: 每个类别的召回率。
    """
    # 加载 YOLO 模型
    model = YOLO("/home/featurize/work/bhintern/ultralytics/runs/detect/train/weights/best.pt")

    # 获取图片路径列表
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)
                   if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # 用于统计所有类别的 TP 和 FN
    class_stats = {}

    for image_path in image_paths:
        # 获取对应的标注文件路径
        label_path = os.path.join(label_folder, os.path.basename(image_path).replace(".jpg", ".txt").replace(".png", ".txt"))

        # 运行推理
        preds = model(image_path)[0]

        # 统计推理结果中每个类别的数量
        predicted_counts = {}
        for cls in preds.boxes.cls.cpu().numpy():
            cls = int(cls)
            if cls in predicted_counts:
                predicted_counts[cls] += 1
            else:
                predicted_counts[cls] = 1

        # 读取标注文件中每个类别的数量
        true_counts = read_label_file(label_path)

        # 更新每个类别的 TP 和 FN
        for cls in set(list(predicted_counts.keys()) + list(true_counts.keys())):
            tp = min(predicted_counts.get(cls, 0), true_counts.get(cls, 0))
            fn = true_counts.get(cls, 0) - tp

            if cls not in class_stats:
                class_stats[cls] = {"tp": 0, "fn": 0}
            class_stats[cls]["tp"] += tp
            class_stats[cls]["fn"] += fn

    # 计算每个类别的召回率
    recalls = {}
    for cls, stats in class_stats.items():
        tp = stats["tp"]
        fn = stats["fn"]
        recalls[cls] = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return recalls

if __name__ == '__main__':
    image_folder = "/home/featurize/work/bhintern/yolodata/valid/images"
    label_folder = "/home/featurize/work/bhintern/yolodata/valid/labels"

    recalls = evaluate_model(image_folder, label_folder)

    # 输出每个类别的召回率
    for cls, recall in recalls.items():
        print(f"Class {cls}: Recall = {recall:.4f}")

    # 输出平均召回率
    avg_recall = sum(recalls.values()) / len(recalls)
    print(f"Average Recall: {avg_recall:.4f}")