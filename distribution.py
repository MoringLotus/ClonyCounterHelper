import json
import os
from tqdm import tqdm
json_fold = '/home/featurize/work/bhintern/data/label'
clony_dict = {}
name_set = set()

for item in os.listdir(json_fold):
    # 构造完整的文件路径
    file_path = os.path.join(json_fold, item)
    
    # 检查是否是文件
    if os.path.isfile(file_path):
        # 打开并读取文件内容
        with open(file_path, 'r') as file:
            contents = json.load(file)  # 使用 json.load() 读取文件内容
            
            # 遍历 labels 列表
            for content in contents['labels']:
                # 将类别名称添加到集合中
                name_set.add(content['class'])
                
                # 更新字典中的计数
                if content['class'] in clony_dict:
                    clony_dict[content['class']] += 1
                else:
                    clony_dict[content['class']] = 1

# 打印结果
for class_name, count in clony_dict.items():
    print(f"类别: {class_name}, 数量: {count}")