"""
题目2参考答案：OpenCV 图像几何变换
实现图像的缩放、旋转和翻转
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')

# 检查图像是否成功读取
if image is None:
    print("错误：无法读取图像 image.jpg")
    print("请确保在当前目录下放置 image.jpg 文件")
    exit(1)

print(f"原始图像大小: {image.shape[1]} x {image.shape[0]}")

# 1. 缩放图像到 50%
scale_percent = 50
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
print(f"缩放后图像大小: {resized_image.shape[1]} x {resized_image.shape[0]}")

# 2. 旋转图像 45 度（逆时针）
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# 获取旋转矩阵
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

# 计算旋转后图像的新边界
cos = np.abs(rotation_matrix[0, 0])
sin = np.abs(rotation_matrix[0, 1])
new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# 调整旋转矩阵的平移部分
rotation_matrix[0, 2] += (new_w / 2) - center[0]
rotation_matrix[1, 2] += (new_h / 2) - center[1]

# 执行旋转
rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
print(f"旋转后图像大小: {rotated_image.shape[1]} x {rotated_image.shape[0]}")

# 3. 水平翻转图像
flipped_image = cv2.flip(image, 1)  # 1 表示水平翻转
print("水平翻转完成")

# 将所有图像从 BGR 转换为 RGB（用于 matplotlib 显示）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
rotated_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
flipped_rgb = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)

# 使用 matplotlib 显示所有图像
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('原始图像', fontsize=14)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(resized_rgb)
plt.title('缩放 (50%)', fontsize=14)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(rotated_rgb)
plt.title('旋转 (45度)', fontsize=14)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(flipped_rgb)
plt.title('水平翻转', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.savefig('transform_result.png', dpi=150, bbox_inches='tight')
print("\n变换结果已保存为 transform_result.png")

# 保存旋转后的图像
cv2.imwrite('rotated_image.jpg', rotated_image)
print("旋转后的图像已保存为 rotated_image.jpg")

print("\n任务完成！")
