"""
题目4参考答案：Matplotlib 数据可视化
绘制阻尼正弦波并保存图像
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. 生成 x 数组：100 个点，范围从 0 到 4π
x = np.linspace(0, 4 * np.pi, 100)

# 2. 计算 y = sin(x) * e^(-0.1*x)（阻尼正弦波）
y = np.sin(x) * np.exp(-0.1 * x)

# 3. 创建图形并绘制曲线（蓝色实线）
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)

# 4. 添加标题
plt.title('Damped Sine Wave', fontsize=16, fontweight='bold')

# 5. 添加坐标轴标签
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

# 6. 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 7. 保存图像
output_file = 'wave.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图像已保存为 {output_file}")

# 可选：显示图像（在实际测试中可以注释掉）
# plt.show()

plt.close()
