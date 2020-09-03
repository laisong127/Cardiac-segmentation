import random

import torch
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np


def imageaug(img_label):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # 左右翻转
        iaa.Flipud(0.5),  # 上下翻转
        iaa.Sometimes(0.3, iaa.Affine(
            rotate=(-10, 10),  # 旋转一定角度
            shear=(-10, 10),  # 拉伸一定角度（矩形变为平行四边形状）
            order=0,  # order=[0, 1],   #使用最邻近差值或者双线性差值
            cval=0,  # cval=(0, 255),  #全白全黑填充
            mode='constant'  # mode=ia.ALL  #定义填充图像外区域的方法
        )),
        # iaa.Crop(percent=(0, 0.1)),
        iaa.Sometimes(0.3, iaa.ElasticTransformation(alpha=(0, 10.0), sigma=(4.0, 6.0))) , # 把像素移动到周围的地方
        iaa.Sometimes(0.3,iaa.GaussianBlur(sigma=0.1)),
        iaa.Sometimes(0.3,iaa.ContrastNormalization(1.1))
    ])
    label = img_label[1]
    imglab_aug = seq.augment_images(img_label)
    img_aug = imglab_aug[0]
    lab_aug = imglab_aug[1]
    lab_aug = np.clip(np.round(lab_aug), np.min(label), np.max(label))
    return img_aug, lab_aug


if __name__ == '__main__':
    img = plt.imread('/home/laisong/MRI2IMG/TRAIN_IMG(A)/ED_A0S9V9_sa_5.png',format='L')
    label = plt.imread('/home/laisong/MRI2IMG/TRAIN_LABEL(A)/ED_A0S9V9_sa_5.png',format='L')
    img_reshape = np.reshape(img, (1,img.shape[0], img.shape[1]))
    label_reshape = np.reshape(label, (1,label.shape[0], label.shape[1]))
    imglab = np.concatenate((img_reshape, label_reshape), axis=0)
    print(imglab.shape)
    img_aug, lab_aug = imageaug(imglab)

    print(np.max(lab_aug))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('img')
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.title('lab')
    plt.imshow(label)
    plt.subplot(2, 2, 3)
    plt.title('img_aug')
    plt.imshow(img_aug)
    plt.subplot(2, 2, 4)
    plt.title('lab_aug')
    plt.imshow(lab_aug)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
