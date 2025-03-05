import numpy as np
from PIL import Image

# 读取图片
img = Image.open("image.png").convert("L")

# 将图片转换为灰度图矩阵
img_array = np.array(img)

# 设置噪声参数
noise_mean = 0
noise_std = 10

# 循环添加噪声并保存图片
for i in range(20):
    # 生成随机噪声
    noise = np.random.normal(noise_mean, noise_std, img_array.shape)

    # 将噪声添加到图片上
    img_array = img_array + noise

    # 将图像模式转换为 L 或 RGB
    img_array = img_array.astype(np.uint8)

    # 保存图片
    Image.fromarray(img_array).save(f"noisy_image_{i}.png")