"""
题目4参考答案：Matplotlib 基础绘图
绘制多条数学函数曲线
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（如果需要显示中文标题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 创建 x 轴数据：0 到 10，100 个点
x = np.linspace(0, 10, 100)

# 2. 计算三组 y 轴数据
y1 = x ** 2  # 二次函数
y2 = x ** 3  # 三次函数
y3 = 2 ** x  # 指数函数

print("数据范围：")
print(f"x 范围: [{x.min():.2f}, {x.max():.2f}]")
print(f"y1 (x²) 范围: [{y1.min():.2f}, {y1.max():.2f}]")
print(f"y2 (x³) 范围: [{y2.min():.2f}, {y2.max():.2f}]")
print(f"y3 (2^x) 范围: [{y3.min():.2f}, {y3.max():.2f}]")

# 3. 创建图形并绘制曲线
plt.figure(figsize=(10, 6))

# 绘制三条曲线
plt.plot(x, y1, color='red', linestyle='-', linewidth=2, label='y = x²')
plt.plot(x, y2, color='blue', linestyle='--', linewidth=2, label='y = x³')
plt.plot(x, y3, color='green', linestyle='-.', linewidth=2, label='y = 2^x')

# 4. 添加标题
plt.title('数学函数对比', fontsize=16, fontweight='bold')

# 5. 添加坐标轴标签
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)

# 6. 添加图例和网格
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# 调整布局
plt.tight_layout()

# 7. 保存图像
output_file = 'functions_plot.png'
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"\n图像已保存为 {output_file}")

# 可选：显示图像
# plt.show()

plt.close()

print("任务完成！")
