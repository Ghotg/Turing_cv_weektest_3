"""
题目1参考答案：OpenCV 图像读取与显示
读取图像、打印信息、转换灰度图、显示和保存
"""

import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image.jpg')

# 检查图像是否成功读取
if image is None:
    print("错误：无法读取图像 image.jpg")
    print("请确保在当前目录下放置 image.jpg 文件")
    exit(1)

# 打印图像基本信息
print("=== 图像基本信息 ===")
print(f"图像形状: {image.shape}")
print(f"高度: {image.shape[0]} 像素")
print(f"宽度: {image.shape[1]} 像素")
if len(image.shape) == 3:
    print(f"通道数: {image.shape[2]}")
else:
    print("通道数: 1 (灰度图)")
print(f"数据类型: {image.dtype}")

# 将图像从 BGR 转换为 RGB（用于 matplotlib 显示）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(f"\n灰度图像形状: {gray_image.shape}")

# 使用 matplotlib 显示图像
plt.figure(figsize=(12, 5))

# 显示原始彩色图像
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('原始彩色图像', fontsize=14)
plt.axis('off')

# 显示灰度图像
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('灰度图像', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.savefig('display_result.png', dpi=150, bbox_inches='tight')
print("\n图像显示结果已保存为 display_result.png")

# 保存灰度图像
cv2.imwrite('gray_image.jpg', gray_image)
print("灰度图像已保存为 gray_image.jpg")

print("\n任务完成！")
