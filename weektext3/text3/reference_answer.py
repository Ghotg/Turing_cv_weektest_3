"""
题目3参考答案：Numpy 数组基础操作
数组创建、索引、切片和统计计算
"""

import numpy as np

# 1. 创建 5x5 数组，元素从 1 到 25
array = np.arange(1, 26).reshape(5, 5)

print("=== 1. 创建的 5x5 数组 ===")
print(array)
print()

# 3. 数组索引和切片
print("=== 3. 数组索引和切片 ===")

# 提取第 2 行（索引为 1）
row_2 = array[1, :]
print(f"第 2 行: {row_2}")

# 提取第 3 列（索引为 2）
col_3 = array[:, 2]
print(f"第 3 列: {col_3}")

# 提取右下角 2x2 子数组
sub_array = array[3:5, 3:5]  # 或 array[-2:, -2:]
print("右下角 2x2 子数组:")
print(sub_array)
print()

# 4. 统计计算
print("=== 4. 统计计算 ===")

# 整个数组的总和
total_sum = np.sum(array)
print(f"数组总和: {total_sum}")

# 每一行的和
row_sums = np.sum(array, axis=1)
print(f"每一行的和: {row_sums}")

# 每一列的和
col_sums = np.sum(array, axis=0)
print(f"每一列的和: {col_sums}")

# 均值和标准差
mean_value = np.mean(array)
std_value = np.std(array)
print(f"数组均值: {mean_value:.2f}")
print(f"数组标准差: {std_value:.2f}")
print()

# 5. 数组转置
transposed_array = array.T  # 或使用 np.transpose(array)
print("=== 5. 转置后的数组 ===")
print(transposed_array)
print()

# 额外验证：转置后的形状
print(f"原数组形状: {array.shape}")
print(f"转置后形状: {transposed_array.shape}")

print("\n任务完成！")
