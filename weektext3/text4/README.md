# 题目4：Matplotlib 数据可视化

## 任务要求

请编写一个Python程序，使用 Matplotlib 完成以下任务：

1. 生成一个包含 100 个点的数组 `x`，范围从 0 到 4π
2. 计算 `y = sin(x) * e^(-0.1*x)`（阻尼正弦波）
3. 使用 Matplotlib 绘制 x 对 y 的图像，使用蓝色实线
4. 添加标题 "Damped Sine Wave"
5. 添加 x 轴标签 "Time"，y 轴标签 "Amplitude"
6. 添加网格线
7. 将图像保存为 `wave.png`

## 要求

- 使用 Numpy 和 Matplotlib 库
- 正确计算阻尼正弦波函数
- 图表要包含所有要求的元素（标题、标签、网格）
- 保存为高质量的 PNG 图像

## 提示

- 使用 `np.linspace()` 生成均匀分布的点
- 使用 `np.sin()` 和 `np.exp()` 计算函数值
- 使用 `plt.plot()` 绘制曲线
- 使用 `plt.title()`, `plt.xlabel()`, `plt.ylabel()` 添加标签
- 使用 `plt.grid()` 添加网格
- 使用 `plt.savefig()` 保存图像
