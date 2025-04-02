from ultralytics import YOLO
import os
import cv2

# 配置参数
MODEL_PATH = "/home/featurize/work/bhintern/ClonyCounterHelper/runs/detect/train2/weights/best.pt"
IMAGE_PATH = "/home/featurize/work/bhintern/ClonyCounterHelper/hard.jpg"
SAVE_DIR = "/home/featurize/work/bhintern/ClonyCounterHelper/output"

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def save_single_prediction(image_path, save_path):
    # 关键参数：关闭置信度显示
    results = model.predict(
        source=image_path,
        save_conf=False,  # 不保存置信度文本
        line_thickness=2  # 可选：调整框线粗细
    )
    
    # 手动绘制无置信度的检测框
    if results and results[0]:
        annotated_frame = results[0].plot(
            boxes=True,      # 显示检测框
            masks=False,     # 不显示分割掩码
            labels=False,    # 不显示类别标签（可选）
            probs=False      # 关键：关闭置信度显示
        )
        cv2.imwrite(save_path, annotated_frame)

# 执行保存
save_single_prediction(IMAGE_PATH, os.path.join(SAVE_DIR, "pred_hard.jpg"))