"""
题目5参考答案：综合应用 - 图像边缘检测与分析
结合 OpenCV、Numpy 和 Matplotlib 进行边缘检测和统计分析
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取图像
image = cv2.imread('image.jpg')

# 检查图像是否成功读取
if image is None:
    print("错误：无法读取图像 image.jpg")
    print("请确保在当前目录下放置 image.jpg 文件")
    exit(1)

print("=== 步骤1：图像读取成功 ===")
print(f"图像大小: {image.shape[1]} x {image.shape[0]}")
print()

# 2. 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("=== 步骤2：已转换为灰度图 ===")

# 3. 高斯模糊预处理
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("=== 步骤3：已应用高斯模糊 (5x5) ===")
print()

# 4. Canny 边缘检测 - 两组不同阈值
edges1 = cv2.Canny(blurred, 50, 150)
edges2 = cv2.Canny(blurred, 100, 200)
print("=== 步骤4：Canny 边缘检测完成 ===")
print("阈值组1: 低阈值=50, 高阈值=150")
print("阈值组2: 低阈值=100, 高阈值=200")
print()

# 5. 使用 Numpy 进行统计分析
total_pixels = gray.shape[0] * gray.shape[1]

# 统计边缘像素数量（非零像素）
edge_pixels1 = np.count_nonzero(edges1)
edge_pixels2 = np.count_nonzero(edges2)

# 计算百分比
percentage1 = (edge_pixels1 / total_pixels) * 100
percentage2 = (edge_pixels2 / total_pixels) * 100

print("=== 步骤5：统计分析结果 ===")
print(f"图像总像素数: {total_pixels:,}")
print()
print("阈值组1 统计:")
print(f"  边缘像素数: {edge_pixels1:,}")
print(f"  边缘像素占比: {percentage1:.2f}%")
print()
print("阈值组2 统计:")
print(f"  边缘像素数: {edge_pixels2:,}")
print(f"  边缘像素占比: {percentage2:.2f}%")
print()

# 6. 使用 Matplotlib 创建 2x2 子图显示
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 左上：原始灰度图
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('原始灰度图', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# 右上：高斯模糊后的图像
axes[0, 1].imshow(blurred, cmap='gray')
axes[0, 1].set_title('高斯模糊 (5x5)', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# 左下：第一组阈值的边缘检测结果
axes[1, 0].imshow(edges1, cmap='gray')
axes[1, 0].set_title(f'Canny边缘检测 (50, 150)\n边缘像素: {edge_pixels1:,} ({percentage1:.2f}%)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].axis('off')

# 右下：第二组阈值的边缘检测结果
axes[1, 1].imshow(edges2, cmap='gray')
axes[1, 1].set_title(f'Canny边缘检测 (100, 200)\n边缘像素: {edge_pixels2:,} ({percentage2:.2f}%)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

# 添加总标题
fig.suptitle('图像边缘检测与分析', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# 7. 保存显示结果
output_display = 'edge_detection_result.png'
plt.savefig(output_display, dpi=200, bbox_inches='tight')
print(f"=== 步骤7：显示结果已保存为 {output_display} ===")

plt.close()

# 8. 保存第一组阈值的边缘检测结果
output_edges = 'edges.jpg'
cv2.imwrite(output_edges, edges1)
print(f"=== 步骤8：边缘检测结果已保存为 {output_edges} ===")

print("\n任务完成！")
print("\n总结:")
print(f"- 较低阈值 (50, 150) 检测到更多边缘细节")
print(f"- 较高阈值 (100, 200) 只保留更强的边缘")
print(f"- 两组结果的边缘像素差异: {abs(edge_pixels1 - edge_pixels2):,} 像素")
