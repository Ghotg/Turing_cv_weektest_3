# 题目5：综合应用 - 物体计数与分析

## 任务描述

本题是一个综合性题目，需要结合 OpenCV、Numpy 和 Matplotlib 完成物体检测、计数和数据分析任务。

## 任务要求

假设你有一张包含硬币的图像 `coins.jpg`（硬币放在简单背景上）。请编写程序完成以下任务：

1. 读取图像 `coins.jpg`
2. 转换为灰度图，并应用二值化阈值处理（使用 Otsu 方法）或 Canny 边缘检测来分离物体
3. 如有必要，使用形态学操作（闭运算/开运算）来清理噪声
4. 使用轮廓检测来计数硬币数量，并打印输出
5. 计算每个轮廓的面积
6. 使用 Matplotlib 绘制硬币面积的直方图，用于分析大小分布
7. 保存处理后的图像（绘制了轮廓的图像）为 `coins_detected.jpg`
8. 保存面积直方图为 `area_hist.png`

## 准备工作

请自行准备一张包含多个硬币的图片，命名为 `coins.jpg`，放在题目文件夹中。

## 要求

- 综合运用 OpenCV、Numpy 和 Matplotlib
- 正确实现图像预处理、物体检测和数据分析
- 能够准确计数和分析物体

## 提示

### OpenCV 部分
- 使用 `cv2.cvtColor()` 转换为灰度图
- 使用 `cv2.threshold()` 配合 `cv2.THRESH_OTSU` 进行阈值处理
- 或使用 `cv2.Canny()` 进行边缘检测
- 使用 `cv2.morphologyEx()` 进行形态学操作
- 使用 `cv2.findContours()` 查找轮廓
- 使用 `cv2.drawContours()` 绘制轮廓
- 使用 `cv2.contourArea()` 计算轮廓面积

### Numpy 部分
- 使用 numpy 数组存储和处理面积数据

### Matplotlib 部分
- 使用 `plt.hist()` 绘制直方图
- 添加适当的标题和标签
