"""
题目3参考答案：Numpy 数组操作
创建随机数组、归一化、计算统计量并保存
"""

import numpy as np

# 1. 创建 100x100 的随机整数数组（0-255）
random_array = np.random.randint(0, 256, size=(100, 100))
print("步骤1：已创建 100x100 随机数组")
print(f"数组形状: {random_array.shape}")
print(f"数据类型: {random_array.dtype}")

# 2. 归一化数组（最小-最大缩放）
min_val = np.min(random_array)
max_val = np.max(random_array)
normalized_array = (random_array - min_val) / (max_val - min_val)
print(f"\n步骤2：数组已归一化")
print(f"原始范围: [{min_val}, {max_val}]")
print(f"归一化后范围: [{np.min(normalized_array):.4f}, {np.max(normalized_array):.4f}]")

# 3. 计算均值和标准差
mean_value = np.mean(normalized_array)
std_value = np.std(normalized_array)
print(f"\n步骤3：统计信息")
print(f"均值 (Mean): {mean_value:.6f}")
print(f"标准差 (Standard Deviation): {std_value:.6f}")

# 4. 保存数组到文件
output_file = 'matrix.npy'
np.save(output_file, normalized_array)
print(f"\n步骤4：归一化数组已保存到 {output_file}")

# 验证保存的文件
loaded_array = np.load(output_file)
print(f"\n验证：成功加载保存的数组，形状为 {loaded_array.shape}")
