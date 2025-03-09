from ultralytics import YOLO
import os

import cv2

# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/runs/detect/train3_10n/weights/best.pt")


# 定义图片文件夹路径
image_folder = "/home/featurize/work/bhintern/perclass/C.albicans"
output_folder = "/home/featurize/work/bhintern/perclass/C.albicans_results_v10"  # 输出文件夹

# 获取文件夹中所有图片路径
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)


# 遍历所有图片
for image_path in image_paths:
    # 对单张图片进行推理
    results = model(image_path)
    
    # 获取原始图像
    
    
    # 遍历每个结果
    for result in results:
        img = result.orig_img.copy()
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        classes = result.boxes.cls
        
        # 获取检测到的边界框
        for i, box in enumerate(boxes):
            # 获取边界框坐标和置信度
            x1, y1, x2, y2 = box[:4].cpu().numpy()  # 使用numpy()获取numpy数组
            score = scores[i].item()  # 获取置信度值
            cls = classes[i].item()  # 获取类别索引
            
            # 绘制边界框和置信度
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, f'{score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 保存结果
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)

print("Results saved to", output_folder)
