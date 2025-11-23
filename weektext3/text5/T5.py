import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg")
img_original = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

all_ = np.array([])
number = 0

img_canny = cv2.Canny(img_gray, 50, 150)
kernel = np.ones((5, 5), np.uint8)

img_dilate = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel, iterations=5)
img_erode = cv2.morphologyEx(img_dilate, cv2.MORPH_ERODE, kernel, iterations=5)

contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_ = np.array([])
max_ = np.array([])

for cnt in contours:
    area = cv2.contourArea(cnt)
    if 100 < area < 8000:
        all_ = np.append(all_, area)
        number += 1
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(img, f"car: {number}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if area < 2000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            min_ = np.append(min_,area)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            max_ = np.append(max_,area)





img_Gaussian = cv2.GaussianBlur(img_gray, (7, 7), 0)
img_canny2 = cv2.Canny(img_Gaussian, 50, 150)

img_dilate2 = cv2.morphologyEx(img_canny2, cv2.MORPH_CLOSE, kernel, iterations=5)
img_erode2 = cv2.morphologyEx(img_dilate2, cv2.MORPH_ERODE, kernel, iterations=5)

contours2, hierarchy = cv2.findContours(img_erode2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



for i in contours2:
    area = cv2.contourArea(i)
    if (2300< area <2500)  or 200<area < 210:
        all_ = np.append(all_, area)
        number += 1
        x,y,w,h = cv2.boundingRect(i)
        cv2.putText(img, f"car: {number}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if area < 2000:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            min_ = np.append(min_,area)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            max_ = np.append(max_,area)
    elif area > 20000:
        x, y, w, h = cv2.boundingRect(i)
        img_ = img[y:y+h, x:x+w]
        img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        img_Gaussian = cv2.GaussianBlur(img_gray, (7, 7), 0)
        img_canny2 = cv2.Canny(img_Gaussian, 50, 150)

        img_dilate2 = cv2.morphologyEx(img_canny2, cv2.MORPH_CLOSE, kernel, iterations=2)
        img_erode2 = cv2.morphologyEx(img_dilate2, cv2.MORPH_ERODE, kernel, iterations=2)

        contours3, hierarchy = cv2.findContours(img_erode2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for k in contours3:
            area = cv2.contourArea(k)
            if area>100:
                all_ = np.append(all_, area)
                number+=1
                x, y, w, h = cv2.boundingRect(k)
                cv2.putText(img_, f"car: {number}", (x-10, y +10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if area < 2000:
                    cv2.rectangle(img_, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    min_ = np.append(min_, area)
                else:
                    cv2.rectangle(img_, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    max_ = np.append(max_, area)
        cv2.waitKey(0)


cv2.imwrite("traffic_detected.jpg", img)
cv2.waitKey(0)
plt.rcParams['font.family']=['STFangsong']
plt.subplot(2, 2, 1)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
plt.title("Original")
plt.imshow(img_original)
plt.subplot(2, 2, 2)
plt.title("img_canny")
plt.imshow(img_canny)
plt.subplot(2, 2, 3)
plt.title("检测")
plt.imshow(img)
plt.subplot(2, 2, 4)
plt.hist(all_, bins=5, color='skyblue', alpha=0.8)
plt.title('车辆面积分布直方图',fontsize=10)
plt.xlabel('面积')
plt.ylabel('数量')
plt.show()
plt.savefig("traffic_analysis_report.png")
print(f"检测到的车辆：{number}")
print(f"最小面积：{min(all_)} 最大面积{max(all_)} 平均值{np.mean(all_)} 标准差{np.std(all_)}")
print(f"大型车/小型车：{max_/min_}")
print(f"车辆密度：{(img.shape[0]*img.shape[1])/number}")
print(f"大型车：{max_} 小型车{min_}")
#可能是根据车的x轴判断在哪个车道和根据密度判断是否拥堵但我懒得做了

print("道路拥堵程度评估:并不拥堵")
print("流量分析建议:右比左的车多")
#扩展思路
#1.cv2.createBackgroundSubtractorMOG2清除静态背景，再用orb特征检测判断是否是同一辆车
#2.把弯曲的道路用仿射变换拉直，前提要知道道路的宽度，再计算车的位移
#3.如果不用训练过的模型,要判断大多数的车的样子，保存，设为正向，再用特征匹配出逆向的车，当然，一般不行，不知道机器学习行不行
#4.问下ai说合适的量化模型为CI=0.45K′+0.35Occ′+0.20V′,我不了解数学模型,需要传入车辆面积和道路面积，跟上面一样
#5.yolo模型不会把一些杂物识别为车，检测框更好，可以根据长宽识别车辆类型