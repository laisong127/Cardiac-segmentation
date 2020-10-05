import os

import cv2
import numpy as np

"""
cv2.Canny(image,            # 输入原图（必须为单通道图）
          threshold1, 
          threshold2,       # 较大的阈值2用于检测图像中明显的边缘
          [, edges[, 
          apertureSize[,    # apertureSize：Sobel算子的大小
          L2gradient ]]])   # 参数(布尔值)：
                              true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                              false：使用L1范数（直接将两个方向导数的绝对值相加）。
"""

import cv2
import numpy as np

img1 = cv2.imread("/home/laisong/MMs2020/MRI2IMG/TRAIN_LABEL(A)/ED_A0S9V9_sa_3.png", 0)
img2 = cv2.imread('/home/laisong/MMs2020/MRI2IMG/TRAIN_LABEL(A)/ED_A0S9V9_sa_5.png',0)

# canny(): 边缘检测
# img1 = cv2.GaussianBlur(img1, (3, 3), 0)
img1_01 = img1/255
img1_01_l1 = (img1_01==1).astype(int)*255
img2_01 = img2/255
img2_01_l1 = (img2_01==1).astype(int)*255
cv2.imwrite('tmp.png',img1_01_l1)
img1_01_l1 = cv2.imread('tmp.png')
cv2.imwrite('tmp.png',img2_01_l1)
img2_01_l1 = cv2.imread('tmp.png')
os.remove('tmp.png')
cv2.imshow('img1l1',img1_01_l1)
cv2.imshow('img2l1',img2_01_l1)
# cv2.waitKey()
canny1 = cv2.Canny(img1_01_l1, 50,80)/255
canny2 = cv2.Canny(img2_01_l1,50,80)/255
ds = np.sum(np.abs(canny1-canny1))#/(np.sum(canny1)+np.sum(canny2)))

print(np.sum(canny1),np.sum(canny2),ds,ds/(np.sum(canny1)+np.sum(canny2)))

# # 形态学：边缘检测
# _, Thr_img = cv2.threshold(img1, 210, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
# gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度
#
# cv2.imshow("img1", img1)
# cv2.imshow("gradient", gradient)
cv2.imshow('Canny1', canny1)
cv2.imshow('Canny2', canny2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import matplotlib.pyplot as plt
# from skimage import measure,data,color
# from scipy.misc import imread
#
# #生成二值测试图像
# img=color.rgb2gray(data.horse())
# # img = imread('/home/laisong/MMs2020/MRI2IMG/TRAIN_LABEL(A)/ED_A0S9V9_sa_3.png')
#
#
# #检测所有图形的轮廓
# contours = measure.find_contours(img, 0.5)
# print(contours[0][:, 1])
#
# #绘制轮廓
# fig, axes = plt.subplots(1,2,figsize=(8,8))
# ax0, ax1= axes.ravel()
# ax0.imshow(img,plt.cm.gray)
# ax0.set_title('original image')
#
# rows,cols=img.shape
# ax1.axis([0,rows,cols,0])
# for n, contour in enumerate(contours):
#     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
# ax1.axis('image')
# ax1.set_title('contours')
# plt.show()

# o1 = cv2.imread('/home/laisong/MMs2020/MRI2IMG/TRAIN_LABEL(A)/ED_A0S9V9_sa_3.png',0)
# # cv2.imshow("original",o1)
# # cv2.waitKey()
# # gray1 = cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
# ret,binary1 = cv2.threshold(o1,0,255,cv2.THRESH_BINARY)
# # cv2.imshow("binary",binary1)
# contours1,hierarchy = cv2.findContours(binary1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# o1=cv2.cvtColor(o1,cv2.COLOR_GRAY2RGB)
# cv2.drawContours(o1, contours1, -1, (0,255,0), 3)
# cv2.imshow('result_all',o1)
#
# o1 = cv2.imread('/home/laisong/MMs2020/MRI2IMG/TRAIN_LABEL(A)/ED_A0S9V9_sa_5.png',0)
# print(type(o1))
# # cv2.imshow("original",o1)
# # cv2.waitKey()
# # gray1 = cv2.cvtColor(o1,cv2.COLOR_BGR2GRAY)
# ret,binary1 = cv2.threshold(o1,0,255,cv2.THRESH_BINARY)
# cv2.imshow("binary",binary1)
# # cv2.waitKey()
# contours2,hierarchy = cv2.findContours(binary1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# print(np.size(contours2))
# o1=cv2.cvtColor(o1,cv2.COLOR_GRAY2RGB)
# # cv2.drawContours(o1, contours2, 0, (0,255,0), 3)
# # cv2.imshow('result1',o1)
# cv2.drawContours(o1, contours2, -1, (0,255,0), 3)
# cv2.imshow('result_all',o1)
# # sd = cv2.createShapeContextDistanceExtractor()
# # #计算距离
# # d1 = sd.computeDistance(contours1,contours2)
# # print("距离 = ",d1)
# d1 = cv2.matchShapes(contours1,contours2,1,0)
# print("距离 = ",d1)
# cv2.waitKey()


