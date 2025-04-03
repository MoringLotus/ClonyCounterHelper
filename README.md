
## 代码结构 ##
可视化
visualable_analysis_pic.py
标签结构转换<br>
transfer_label_yolo.py
测试结果<br>
test_recall.py
图像分布测试<br>
distribution.py
训练代码<br>
ultralytics 目录<br>
  -few_shot_run.py：小样本训练<br>
  -run.py：全样本训练<br>
  -test.py：测试代码<br>
  -judge_the_result.py 结果分析，指标生成代码<br>

## 使用方法 ##
环境准备： python >= 3.10.10 pip install ultralytics
python run.py 执行训练代码，内部修改配置的yaml文件， 结果会输出到run文件夹内部
python test.py 测试训练权重的效果
python test_recall.py 测试相应模型权重的召回率
