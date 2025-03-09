import cv2
import numpy as np

def preprocess_image(image, I0):
    """
    使用对数变换预处理图像
    :param image: 输入图像
    :param I0: 参考亮度值
    :return: 预处理后的图像
    """
    # 将图像转换为浮点数进行计算
    image = image.astype(np.float32)
    
    # 计算对数变换
    log_Ii = np.log10(image + 1)  # 加1避免对数为负无穷
    log_I0 = np.log10(I0 + 1)
    
    # 应用公式 lgIi′ = lgI0 − lgIi
    log_Ii_prime = log_I0 - log_Ii
    
    # 转换回原始范围
    Ii_prime = 10 ** log_Ii_prime - 1  # 减1以抵消之前的加1操作
    
    # 确保像素值在0到255之间
    Ii_prime = np.clip(Ii_prime, 0, 255)
    
    # 转换回uint8类型
    Ii_prime = Ii_prime.astype(np.uint8)
    
    return Ii_prime

# 读取图像
image_path = '/home/featurize/work/bhintern/data/img/309.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图像

# 检查图像是否成功加载
if image is None:
    print("无法加载图像，请检查路径是否正确。")
    exit()

# 计算参考亮度值 I0，这里以图像的平均亮度为例
I0 = np.mean(image)

# 预处理图像
preprocessed_image = preprocess_image(image, I0)

# 保存预处理后的图像
cv2.imwrite('preprocessed_image.jpg', preprocessed_image)

print("预处理后的图像已保存到当前文件夹。")