from ultralytics import YOLO
epo = 100
# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/yolov5nu.pt")  # 使用 YOLOv10 的预训练模型
# train2 是使用YOLO11x
# train1 是使用YOLO11n
# 训练模型
results = model.train(
    data="/home/featurize/work/bhintern/ultralytics/ultralytics/cfg/datasets/AGAR.yaml",  # 数据集配置文件路径
    epochs=epo,        # 训练轮数
    imgsz=640,         # 图像大小
    batch=16,          # 批量大小
    device="cuda"      # 使用 GPU 设备（如果有）
)



from ultralytics import YOLO
# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/yolo11n.pt")  # 使用 YOLOv10 的预训练模型

# 训练模型
results = model.train(
    data="/home/featurize/work/bhintern/ultralytics/ultralytics/cfg/datasets/AGAR.yaml",  # 数据集配置文件路径
    epochs=epo,        # 训练轮数
    imgsz=640,         # 图像大小
    batch=16,          # 批量大小
    device="cuda"      # 使用 GPU 设备（如果有）
)



from ultralytics import YOLO
# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/yolov10n.pt")  # 使用 YOLOv10 的预训练模型

# 训练模型
results = model.train(
    data="/home/featurize/work/bhintern/ultralytics/ultralytics/cfg/datasets/AGAR.yaml",  # 数据集配置文件路径
    epochs=epo,        # 训练轮数
    imgsz=640,         # 图像大小
    batch=16,          # 批量大小
    device="cuda"      # 使用 GPU 设备（如果有）
)







