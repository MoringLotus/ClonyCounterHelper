from ultralytics import YOLO
import os
import cv2
import numpy as np
import tqdm
# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/runs/detect/train_v5nu/weights/best.pt")

# 定义图片文件夹路径
image_folder = "/home/featurize/work/bhintern/perclass/C.albicans"
output_folder = "/home/featurize/work/bhintern/perclass/C.albicans_results_patches"  # 输出文件夹

# 获取文件夹中所有图片路径
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 瓦片算法参数
tile_size = (512, 512)  # 瓦片大小
overlap = 0.25  # 瓦片重叠比例

# 遍历所有图片
for image_path in image_paths:
    # 加载原始图像
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 初始化用于保存检测结果的列表
    all_boxes = []
    all_scores = []
    all_classes = []

    # 计算瓦片的起始位置
    tile_h, tile_w = tile_size
    overlap_pixels = int(tile_h * overlap)
    stride_h = tile_h - overlap_pixels
    stride_w = tile_w - overlap_pixels

    # 遍历图像的每个瓦片
    for y in range(0, h - tile_h + overlap_pixels, stride_h):
        for x in range(0, w - tile_w + overlap_pixels, stride_w):
            # 提取瓦片
            tile = img[y:y + tile_h, x:x + tile_w]

            # 对瓦片进行目标检测
            results = model(tile)

            # 遍历检测结果
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框
                scores = result.boxes.conf.cpu().numpy()  # 获取置信度
                classes = result.boxes.cls.cpu().numpy()  # 获取类别索引

                # 将边界框坐标调整到原始图像的坐标系
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = int(x1) + x, int(y1) + y, int(x2) + x, int(y2) + y
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(score)
                    all_classes.append(cls)

    # 绘制所有检测结果
    for box, score, cls in zip(all_boxes, all_scores, all_classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 保存结果
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

print("Results saved to", output_folder)