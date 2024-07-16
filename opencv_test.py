import cv2
import numpy as np

# 读取图片
image = cv2.imread('./imgs/1.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值处理提取高亮部分
_, thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

# 对高亮部分应用高斯模糊以创建发光效果
glow = cv2.GaussianBlur(thresh, (0, 0), sigmaX=30, sigmaY=30)

# 将发光效果转换为彩色图像
glow_color = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)

# 调整发光效果的亮度
glow_color = (glow_color * 0.5).astype(np.uint8)

print(glow_color.shape)
print(image.shape)

# 将发光效果叠加到原始图片上
result = cv2.addWeighted(image, 1, glow_color, 1, 0)

# # 显示结果
cv2.imshow('origin img', image)
cv2.imshow('Glow Effect', result)
cv2.imshow('thresh', thresh)
# cv2.imshow('glow', glow)
# cv2.imshow('glow_color', glow_color)

cv2.waitKey(0)
cv2.destroyAllWindows()

###发光特效实验