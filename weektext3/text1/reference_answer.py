"""
题目1参考答案：OpenCV 图像基础处理
读取图像、转换为灰度图、应用高斯模糊并保存
"""

import cv2
import os

# 定义图像路径
image_path = 'images/image.jpg'
output_path = 'processed.jpg'

# 读取图像
image = cv2.imread(image_path)

# 检查图像是否成功读取
if image is None:
    print(f"错误：无法读取图像 {image_path}")
    print("请确保在当前目录下创建 images 文件夹，并放置 image.jpg 文件")
    exit(1)

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊（核大小为 5x5）
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 保存处理后的图像
cv2.imwrite(output_path, blurred_image)

print(f"图像处理完成！结果已保存为 {output_path}")
