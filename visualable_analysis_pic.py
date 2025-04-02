from ultralytics import YOLO
import os
import cv2
from PIL import Image, ImageDraw

# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ClonyCounterHelper/ultralytics/runs/detect/train_v10_1028/weights/best.pt")

# 定义图片文件夹路径和输出文件夹路径
image_folder = "/home/featurize/work/bhintern/yolodata/valid/images"
output_folder = "/home/featurize/work/bhintern/yolodata/valid/output"  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图片路径
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png', '.jpeg'))]

# 类别 ID 与颜色的映射（可以根据需要自定义颜色）
class_colors = {
    0: (255, 0, 0),    # 红色
    1: (0, 255, 0),    # 绿色
    2: (0, 0, 255),    # 蓝色
    3: (255, 255, 0),  # 黄色
    4: (255, 0, 255),  # 品红色
}

# 遍历所有图片
for image_path in image_paths:
    # 对单张图片进行推理
    results = model.predict(image_path)
    
    # 获取原始图像
    orig_img = results[0].orig_img
    
    # 将原始图像转换为PIL图像
    pil_img = Image.fromarray(orig_img)
    
    # 创建一个绘图对象
    draw = ImageDraw.Draw(pil_img)
    
    # 遍历检测结果
    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        # 获取类别ID和对应的颜色
        cls_id = int(results[0].boxes.cls[i])
        color = class_colors.get(cls_id, (255, 255, 255))  # 默认为白色
        
        # 绘制边界框（线条细一点）
        draw.rectangle(box, outline=color, width=1)  # width=1 表示线条宽度为1
    
    # 保存标注后的图像
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    pil_img.save(output_image_path)  # 使用Pillow保存图像
    
    # 如果需要，也可以显示图像
    # pil_img.show()

print("All images have been processed and saved.")