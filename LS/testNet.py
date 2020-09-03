from torch.autograd import Variable
import nibabel as nib
import numpy as np
from PIL import Image

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

from metrics import dice_coeff
import torch
from Newunet import Insensee_3Dunet
import MRIdataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def saveresult(test_loader, model):
    i = 0
    for img, label in test_loader:
        img = img.to(device)
        pred = torch.argmax(model(img), 1)
        pred = pred.cpu()

        pred = torch.squeeze(pred)
        pred = pred.numpy().astype(float)
        pred = pred.transpose(1, 2, 0)

        label = torch.squeeze(label)
        label = label.numpy().astype(float)
        label = label.transpose(1, 2, 0)

        new_image = nib.Nifti1Image(pred, np.eye(4))
        nib.save(new_image, r'/home/peng/Desktop/CROP/pred/pred_%d.nii.gz' % i)

        new_label = nib.Nifti1Image(label, np.eye(4))
        nib.save(new_label, r'/home/peng/Desktop/CROP/pred/label_%d.nii.gz' % i)

        i += 1


def test_model():
    MAX = 0
    save = 0
    # for index in range(100):
    model = Insensee_3Dunet(1).to(device)
    model.load_state_dict(torch.load('./3dunet_model_save/weights_390.pth'))
    test_dataset = MRIdataset.LiverDataset(MRIdataset.test_imagepath, MRIdataset.test_labelpath,
                                           MRIdataset.testimg_ids, MRIdataset.testlabel_ids)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    train_dataset = MRIdataset.LiverDataset(MRIdataset.imagepath, MRIdataset.labelpath,
                                           MRIdataset.img_ids, MRIdataset.label_ids)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)



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

    for img, label in test_loader:
        img = img.to(device)
        pred = torch.argmax(model(img), 1)
        pred = pred.cpu()

        LV_dice, LV_jac, RV_dice, RV_jac, Myo_dice, Myo_jac = dice_coeff(pred, label)

        LV_Dice += LV_dice
        LV_Jac += LV_jac
        RV_Dice += RV_dice
        RV_Jac += RV_jac
        Myo_Dice += Myo_dice
        Myo_Jac += Myo_jac

        print('LV_Dice_%d:' % i, '%.6f' % LV_dice, '||', 'RV_Dice_%d:' % i, '%.6f' % RV_dice, '||'
              , 'Myo_Dice_%d:' % i, '%.6f' % Myo_dice)

        i += 1
    print('===============================================')
    print('LV_Dice_avg:', LV_Dice / i, 'RV_Dice_avg:', RV_Dice / i, 'Myo_Dice_avg:', Myo_Dice / i)
    # if (LV_Dice/i+RV_Dice/i+Myo_Dice/i) > MAX:
    #     MAX = LV_Dice/i+RV_Dice/i+Myo_Dice/i
    #     save = index


if __name__ == '__main__':
    test_model()

    # img = nib.load(r'/home/peng/Desktop/CROP/pred/pred_2.nii.gz').get_data()
    # img = img.flatten()
    # for i in range(len(img)):
    #     if img[i]!=0:
    #         print(img[i])
