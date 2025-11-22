# 题目2：OpenCV 图像几何变换

## 任务要求

请编写一个Python程序，对图像进行基本的几何变换操作：

1. 读取一张图像文件 `image.jpg`
2. 对图像进行以下变换：
   - 缩放：将图像缩小到原来的 50%
   - 旋转：将原图像逆时针旋转 45 度（不裁剪，保留完整内容）
   - 翻转：将原图像进行水平翻转
3. 使用 matplotlib 显示四张图像（原图、缩放、旋转、翻转），使用 2x2 的子图布局
4. 将旋转后的图像保存为 `rotated_image.jpg`

## 准备工作

请准备一张测试图片 `image.jpg`，放在题目文件夹中。

## 要求

- 使用 OpenCV (cv2) 库完成图像变换
- 使用 Matplotlib 显示结果
- 代码要有清晰的注释

## 提示

- 使用 `cv2.resize()` 进行图像缩放
- 使用 `cv2.getRotationMatrix2D()` 和 `cv2.warpAffine()` 进行旋转
- 使用 `cv2.flip()` 进行图像翻转
- 记得将 BGR 转换为 RGB 用于 matplotlib 显示
