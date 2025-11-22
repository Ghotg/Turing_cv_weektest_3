"""
题目2参考答案：OpenCV 颜色检测与轮廓识别
检测红色区域、创建掩码、找到轮廓并绘制边界框
"""

import cv2
import numpy as np

# 定义图像路径
image_path = 'image.jpg'
mask_output_path = 'red_mask.jpg'
result_output_path = 'result.jpg'

# 读取图像
image = cv2.imread(image_path)

# 检查图像是否成功读取
if image is None:
    print(f"错误：无法读取图像 {image_path}")
    print("请确保在当前目录下放置包含红色物体的 image.jpg 文件")
    exit(1)

# 转换为 HSV 颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义红色的 HSV 范围
# 红色在 HSV 中有两个范围（因为红色在色相环的两端）
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 创建两个掩码并合并
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# 保存掩码图像
cv2.imwrite(mask_output_path, red_mask)
print(f"红色掩码已保存为 {mask_output_path}")

# 查找轮廓
contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图像上绘制边界框
result_image = image.copy()
for contour in contours:
    # 获取边界框坐标
    x, y, w, h = cv2.boundingRect(contour)
    # 绘制绿色矩形（BGR格式）
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 保存结果图像
cv2.imwrite(result_output_path, result_image)
print(f"检测到 {len(contours)} 个红色区域")
print(f"结果图像已保存为 {result_output_path}")
