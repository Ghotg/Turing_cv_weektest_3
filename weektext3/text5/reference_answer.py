"""
题目5参考答案：综合应用 - 物体计数与分析
结合 OpenCV、Numpy 和 Matplotlib 进行硬币检测、计数和分析
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持（可选）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义文件路径
image_path = 'coins.jpg'
output_image_path = 'coins_detected.jpg'
output_hist_path = 'area_hist.png'

# 1. 读取图像
image = cv2.imread(image_path)

# 检查图像是否成功读取
if image is None:
    print(f"错误：无法读取图像 {image_path}")
    print("请确保在当前目录下放置 coins.jpg 文件")
    exit(1)

print("步骤1：图像读取成功")

# 2. 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("步骤2：已转换为灰度图")

# 应用高斯模糊减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用 Otsu 方法进行二值化
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("步骤2：已应用二值化阈值处理（Otsu方法）")

# 3. 形态学操作清理噪声
# 使用闭运算填充物体内部的小孔
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
# 使用开运算去除小的噪声点
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
print("步骤3：已应用形态学操作清理噪声")

# 4. 查找轮廓
contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"步骤4：检测到 {len(contours)} 个轮廓")

# 5. 计算每个轮廓的面积，并过滤掉太小的轮廓
min_area = 100  # 最小面积阈值，过滤噪声
areas = []
valid_contours = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        areas.append(area)
        valid_contours.append(contour)

areas = np.array(areas)
coin_count = len(valid_contours)
print(f"步骤5：过滤后检测到 {coin_count} 个有效物体（硬币）")
print(f"面积统计 - 最小: {np.min(areas):.2f}, 最大: {np.max(areas):.2f}, 平均: {np.mean(areas):.2f}")

# 7. 在原图上绘制轮廓
result_image = image.copy()
cv2.drawContours(result_image, valid_contours, -1, (0, 255, 0), 2)

# 在图像上标注硬币数量
text = f'Coins: {coin_count}'
cv2.putText(result_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 0, 255), 2, cv2.LINE_AA)

# 保存结果图像
cv2.imwrite(output_image_path, result_image)
print(f"步骤7：已保存检测结果图像到 {output_image_path}")

# 6 & 8. 使用 Matplotlib 绘制面积直方图
plt.figure(figsize=(10, 6))
plt.hist(areas, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('硬币面积分布直方图', fontsize=16, fontweight='bold')
plt.xlabel('面积 (像素²)', fontsize=12)
plt.ylabel('数量', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 添加统计信息
stats_text = f'总数: {coin_count}\n平均面积: {np.mean(areas):.1f}\n标准差: {np.std(areas):.1f}'
plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_hist_path, dpi=300, bbox_inches='tight')
print(f"步骤8：已保存面积直方图到 {output_hist_path}")

plt.close()

print("\n所有任务完成！")
