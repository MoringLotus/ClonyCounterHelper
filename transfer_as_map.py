import os
import json
from PIL import Image

def convert_json_to_yolo(json_folder, image_folder, output_folder):
    """
    将 JSON 格式的标注转换为 YOLO 格式的标注文件，并按照指定的类别顺序分配编号。
    
    参数:
        json_folder: 存放 JSON 文件的文件夹路径。
        image_folder: 存放图像文件的文件夹路径。
        output_folder: YOLO 标注文件输出的文件夹路径。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 如果输出文件夹不存在，则创建

    # 定义固定的类别和编号映射
    class_to_id = {
        "C.albicans": 0,
        "B.subtilis": 1,
        "S.aureus": 2,
        "E.coli": 3,
        "P.aeruginosa": 4,
        "Contamination": 5,
        "Defect": 6
    }

    # 遍历 JSON 文件夹中的所有 JSON 文件
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_folder, json_file)
            print(f"处理文件: {json_file}")

            # 读取 JSON 文件
            with open(json_path, "r") as f:
                data = json.load(f)

            # 获取对应的图像文件名
            image_file = json_file.replace(".json", ".jpg")  # 假设图像格式为 .jpg
            image_path = os.path.join(image_folder, image_file)

            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"未找到对应的图像文件: {image_path}")
                continue

            # 使用 Pillow 获取图像的宽度和高度
            with Image.open(image_path) as img:
                image_width, image_height = img.size
                print(f"图像宽度: {image_width}, 图像高度: {image_height}")

            # 遍历 JSON 中的标注信息
            yolo_labels = []  # 存储当前文件的 YOLO 标注信息
            for label in data["labels"]:
                class_name = label["class"]
                if class_name not in class_to_id:
                    raise ValueError(f"未知类别: {class_name}。请检查类别是否正确或是否在类别列表中。")

                class_id = class_to_id[class_name]
                x, y, width, height = label["x"], label["y"], label["width"], label["height"]

                # 归一化坐标
                x_center = (x + width / 2) / image_width
                y_center = (y + height / 2) / image_height
                width_normalized = width / image_width
                height_normalized = height / image_height

                # YOLO 格式：class_id x_center y_center width height
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width_normalized:.6f} {height_normalized:.6f}")

            # 保存 YOLO 标注文件
            output_file = os.path.join(output_folder, json_file.replace(".json", ".txt"))
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_labels))

            print(f"YOLO 标注文件已保存到: {output_file}")

    # 输出类别和编号的对应关系
    print("\n类别和编号的对应关系：")
    for class_name, class_id in class_to_id.items():
        print(f"{class_name}: {class_id}")

    # 可选：将类别和编号的对应关系保存到文件
    classes_file = os.path.join(output_folder, "classes.txt")
    with open(classes_file, "w") as f:
        for class_name in sorted(class_to_id, key=class_to_id.get):
            f.write(f"{class_name}\n")

    print(f"\n类别和编号的对应关系已保存到: {classes_file}")

# 设置文件夹路径
json_folder = "/home/featurize/work/bhintern/yolodata/test/labels"  # JSON 文件所在的文件夹路径
image_folder = "/home/featurize/work/bhintern/yolodata/test/images"  # 图像文件所在的文件夹路径
output_folder = "/home/featurize/work/bhintern/yolodata/test/yololabels"  # 输出 YOLO 标注文件的文件夹路径

# 调用函数
convert_json_to_yolo(json_folder, image_folder, output_folder)