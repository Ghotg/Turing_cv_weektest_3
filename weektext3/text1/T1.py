import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")
print(f"高度：{img.shape[0]} 宽度：{img.shape[1]} 通道数：{img.shape[2]}")
print(f"数据类型：{type(img)}")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image.jpg", gray_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.imshow(gray_image, cmap="gray")
plt.show()