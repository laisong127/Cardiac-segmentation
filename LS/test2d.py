from torch.autograd import Variable
import nibabel as nib
import numpy as np
from PIL import Image

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

from ResNetUNet import ResNetUNet
from metrics import dice_coeff
import torch
from Newunet import Insensee_3Dunet
import MRIdataset
from torch.utils.data import DataLoader
from advanced_model import CleanU_Net
from advanced_model import DeepSupervision_U_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def saveresult2d(test_loader, model):
    i = 0
    for img, label, _, labelfile in test_loader:
        if i<50:
            label3d = []
            img = torch.squeeze(img)
            label = torch.squeeze(label)
            for z in range(img.shape[0]):
                img_2d = img[z, :, :]
                label_2d = label[z, :, :]
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = img_2d.to(device)
                output = model(img_2d)
                # print(output.shape)
                pred = torch.argmax(output, 1)
                # print(pred.shape, label.shape)
                pred = pred.cpu()
                # print(type(pred))
                # label3d.append(pred)
                if z == 0:
                    label3d = pred
                else:
                    label3d = torch.cat((label3d, pred),dim=0)
            print(label3d.shape)
        label = label3d.numpy().astype(float)
        label1 = label.transpose(1, 2, 0)

        new_image = nib.Nifti1Image(label1, np.eye(4))
        nib.save(new_image, '../saveresult/pred_{}'.format(labelfile[0]))

        i += 1


def test_model():
    MAX = 0
    save = 0
    # for index in range(100):
    # model = DeepSupervision_U_Net(1, 4).to(device)
    model = ResNetUNet(4).to(device)
    # print(model)
    model.load_state_dict(torch.load('/home/laisong/MMs/LS/3dunet_model_save/BEST_weights_150.pth'))
    test_dataset = MRIdataset.LiverDataset(MRIdataset.test_imagepath, MRIdataset.test_labelpath,
                                           MRIdataset.testimg_ids, MRIdataset.testlabel_ids,False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    train_dataset = MRIdataset.LiverDataset(MRIdataset.imagepath, MRIdataset.labelpath,
                                            MRIdataset.img_ids, MRIdataset.label_ids,False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    B_dataset = MRIdataset.LiverDataset(MRIdataset.B_imagepath, MRIdataset.B_labelpath,
                                           MRIdataset.Bimg_ids, MRIdataset.Blabel_ids, False)

    B_loader = DataLoader(B_dataset, batch_size=1, shuffle=False, num_workers=4)

    # train_iter = enumerate(train_loader)
    model.eval()
    LV_Dice = 0
    LV_Jac = 0
    RV_Dice = 0
    RV_Jac = 0
    Myo_Dice = 0
    Myo_Jac = 0
    i = 0

    # _, batch = train_iter.__next__()
    # _, batch = train_iter.__next__()
    # img, label= batch
    # i = 1
    # # print(img.shape)
    # # img =torch.squeeze(img)
    # # img = img.numpy().astype(float)
    # # img = img.transpose(1, 2, 0)
    # # new_image = nib.Nifti1Image(img, np.eye(4))
    # # nib.save(new_image, r'/home/peng/Desktop/CROP/train2test/trainimg_%d.nii.gz' % i)
    #
    # img = img.to(device)
    # pred = torch.argmax(model(img), 1)
    # pred = pred.cpu()
    # pred = torch.squeeze(pred)
    # pred = pred.numpy().astype(float)
    # pred = pred.transpose(1, 2, 0)
    #
    # label = torch.squeeze(label)
    # label = label.numpy().astype(float)
    # label = label.transpose(1, 2, 0)
    #
    #
    # new_image = nib.Nifti1Image(pred, np.eye(4))
    # nib.save(new_image, r'/home/peng/Desktop/CROP/train2test/pred_%d.nii.gz' % i)
    #
    # new_label = nib.Nifti1Image(label, np.eye(4))
    # nib.save(new_label, r'/home/peng/Desktop/CROP/train2test/label_%d.nii.gz' % i)

    for img, label,_,_ in test_loader:
        if i<50:
            img = torch.squeeze(img)
            label = torch.squeeze(label)
            LV_dice = 0
            RV_dice = 0
            Myo_dice = 0
            for z in range(img.shape[0]):
                img_2d = img[z, :, :]
                label_2d = label[z, :, :]
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = torch.unsqueeze(img_2d, 0)
                img_2d = img_2d.to(device)
                output = model(img_2d)
                # print(output.shape)
                pred = torch.argmax(output, 1)
                # print(pred.shape, label.shape)
                pred = pred.cpu()

                LV_dice_2d, LV_jac_2d, RV_dice_2d, RV_jac_2d, Myo_dice_2d, Myo_jac_2d = dice_coeff(pred, label_2d)
                LV_dice += LV_dice_2d
                RV_dice += RV_dice_2d
                Myo_dice += Myo_dice_2d

            LV_dice /= img.shape[0]
            RV_dice /= img.shape[0]
            Myo_dice /= img.shape[0]

            LV_Dice += LV_dice
            RV_Dice += RV_dice
            Myo_Dice += Myo_dice

            print('LV_Dice_%d:' % i, '%.6f' % LV_dice, '||', 'RV_Dice_%d:' % i, '%.6f' % RV_dice, '||'
                  , 'Myo_Dice_%d:' % i, '%.6f' % Myo_dice)

            i += 1
    print('===============================================')
    print('LV_Dice_avg:', LV_Dice / i, 'RV_Dice_avg:', RV_Dice / i, 'Myo_Dice_avg:', Myo_Dice / i)
    # if (LV_Dice/i+RV_Dice/i+Myo_Dice/i) > MAX:
    #     MAX = LV_Dice/i+RV_Dice/i+Myo_Dice/i
    #     save = index


if __name__ == '__main__':
    # model = CleanU_Net(1, 4).to(device)
    # model.load_state_dict(torch.load('./3dunet_model_save/weights_200.pth'))
    # test_dataset = MRIdataset.LiverDataset(MRIdataset.test_imagepath, MRIdataset.test_labelpath,
    #                                        MRIdataset.testimg_ids, MRIdataset.testlabel_ids, False)
    #
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # saveresult2d(test_loader, model)
    test_model()

    # img = nib.load(r'/home/peng/Desktop/CROP/pred/pred_2.nii.gz').get_data()
    # img = img.flatten()
    # for i in range(len(img)):
    #     if img[i]!=0:
    #         print(img[i])
