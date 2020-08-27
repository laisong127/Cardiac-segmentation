import torch
import torch.utils.data as data
import os
import numpy as np
import PIL.Image as Image
import nibabel as nib
from scipy.ndimage import rotate
from torch.utils.data import DataLoader
from torchvision import transforms

imagepath = r'../Different_Vendor/TRAIN_A'
labelpath = r'../Different_Vendor/TRAIN_LABEL_A'
imgid = r'../Different_Vendor/Img_A_train.txt'
labelid = r'../Different_Vendor/Label_A_train.txt'
img_ids = [i_id.strip() for i_id in open(imgid)]
label_ids = [i_id.strip() for i_id in open(labelid)]

test_imagepath = r'../Different_Vendor/TEST_A'
test_labelpath = r'../Different_Vendor/TEST_LABEL_A'
imgid = r'../Different_Vendor/Img_A_test.txt'
labelid = r'../Different_Vendor/Label_A_test.txt'
testimg_ids = [i_id.strip() for i_id in open(imgid)]
testlabel_ids = [i_id.strip() for i_id in open(labelid)]

B_imagepath = r'../Different_Vendor/B'
B_labelpath = r'../Different_Vendor/LABEL_B'
imgid = r'../Different_Vendor/Img_B.txt'
labelid = r'../Different_Vendor/Label_B.txt'
Bimg_ids = [i_id.strip() for i_id in open(imgid)]
Blabel_ids = [i_id.strip() for i_id in open(labelid)]


# print(img_ids)

# data.Dataset:
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)
def transform(img):
    p1 = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
    degree = np.random.choice([-90, 0, 180, 90], p=p1.ravel())
    print(degree)
    img = rotate(img, degree, (2, 3), reshape=False, cval=0)
    return img


def normalization(data):
    data = np.asarray(data).squeeze()
    max, min = np.max(data), np.min(data)
    data = (data - min) / (max - min)
    return data


class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, imagepath, labelpath, imgids, labelids, aug):  # root表示图片路径
        self.aug = aug
        n = len(os.listdir(imagepath))  # os.listdir(path)返回指定路径下的文件和文件夹列表。/是真除法,//对结果取整
        imgdirs = os.listdir(imagepath)
        labeldirs = os.listdir(labelpath)
        imgs = []
        labels = []
        imgfile = []
        labelfile = []

        for j, imfile in enumerate(imgdirs):
            # if (file != 'desktop.ini'):
            image = nib.load(imagepath + '/' + imgids[j]).get_data()
            imgs.append(image)

        for j, lafile in enumerate(labeldirs):
            # if (file != 'desktop.ini'):
            label = nib.load(labelpath + '/' + labelids[j]).get_data()
            labels.append(label)

        self.imgs = imgs
        self.labels = labels
        self.imgfile = imgids
        self.labelfile = labelids

    def __getitem__(self, index):
        imgfile = self.imgfile[index]
        labelfile = self.labelfile[index]
        # print(imgfile,labelfile)
        image = self.imgs[index]
        image3 = normalization(image)  # normalization
        label = self.labels[index]
        image4 = torch.from_numpy(image3.transpose((2, 0, 1)))
        label1 = torch.from_numpy(label.transpose((2, 0, 1)))
        if not self.aug:
            image5 = torch.unsqueeze(image4, 0)
            return image5, label1, imgfile, labelfile  # 返回的是图片
        else:
            imgandlabel = [image3.transpose((2, 0, 1)), label.transpose((2, 0, 1))]
            imgandlabel = np.array(imgandlabel)
            # print(imgandlabel.shape)
            imgandlabel = transform(imgandlabel)
            transimg = imgandlabel[0]
            translabel = imgandlabel[1]
            transimg1 = torch.unsqueeze(torch.from_numpy(transimg), 0)
            # print(imgandlabel.shape, imgandlabel[0].shape, imgandlabel[1].shape)
            return transimg1, translabel, imgfile, labelfile  # 返回的是tensor

    def __len__(self):
        return len(self.imgs)  # 400,list[i]有两个元素，[img,mask]


if __name__ == '__main__':

    liver_dataset = LiverDataset(test_imagepath, test_labelpath, testimg_ids, testlabel_ids, False)
    dataloader = DataLoader(liver_dataset, batch_size=1, shuffle=False, num_workers=1)
    i = 0
    for img, label, imgfile, labelfile in dataloader:
        print(img.shape, label.shape, imgfile, labelfile)
        # s1 = ''.join(imgfile)
        # s2 = ''.join(labelfile)
        # img = torch.squeeze(img)
        # img = img.numpy().astype(float)
        # img = img.transpose(1, 2, 0)
        # img90 = rotate(img,90,(0,1),reshape=False,cval=0)
        # new_image = nib.Nifti1Image(img90, np.eye(4))
        # nib.save(new_image, r'/home/yjj/MMs/LS/rotate90/90_'+s1)
        # label = torch.squeeze(label)
        # label = label.numpy().astype(float)
        # label = label.transpose(1, 2, 0)
        # label90 = rotate(label, 90, (0, 1), reshape=False, cval=0)
        # new_label = nib.Nifti1Image(label90, np.eye(4))
        # nib.save(new_label, r'/home/yjj/MMs/LS/rotate90label/90_' + s2)
        i += 1
    print(i)

    # dataiter = enumerate(dataloader)
    # _, batch = dataiter.__next__()
    # images, labels = batch
    # # print(images.shape)
    # images = torch.squeeze(images)
    # # print(images.shape)
    # # print(torch.max(images, 2))
    # label_1 = images[1,:,:]
    # for i in range(192):
    #     for j in range(192):
    #         if label_1[i][j]<0:
    #             print(label_1[i][j])
