from ultralytics import YOLO
# 加载预训练模型
model = YOLO("/home/featurize/work/bhintern/ultralytics/yolov10n.pt")  # 使用 YOLOv10 的预训练模型
# train4 train5 train6
# 训练模型
# results = model.train(
#     data="/home/featurize/work/bhintern/ClonyCounterHelper/ultralytics/ultralytics/cfg/datasets/AGAR_100.yaml",  # 数据集配置文件路径
#     epochs=100,        # 训练轮数
#     imgsz=1280,         # 图像大小
#     batch=16,          # 批量大小
#     device="cuda"      # 使用 GPU 设备（如果有）
# )

# results = model.train(
#     data="/home/featurize/work/bhintern/ClonyCounterHelper/ultralytics/ultralytics/cfg/datasets/AGAR_500.yaml",  # 数据集配置文件路径
#     epochs=100,        # 训练轮数
#     imgsz=1280,         # 图像大小
#     batch=16,          # 批量大小
#     device="cuda"      # 使用 GPU 设备（如果有）
# )

results = model.train(
    data="/home/featurize/work/bhintern/ClonyCounterHelper/ultralytics/ultralytics/cfg/datasets/AGAR_1000.yaml",  # 数据集配置文件路径
    epochs=100,        # 训练轮数
    imgsz=1280,         # 图像大小
    batch=16,          # 批量大小
    device="cuda"      # 使用 GPU 设备（如果有）
)


