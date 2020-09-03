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
from advanced_model import CleanU_Net
from advanced_model import DeepSupervision_U_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def val_model(model):
    test_dataset = MRIdataset.LiverDataset(MRIdataset.test_imagepath, MRIdataset.test_labelpath,
                                           MRIdataset.testimg_ids, MRIdataset.testlabel_ids, False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model.eval()
    LV_Dice = 0
    RV_Dice = 0
    Myo_Dice = 0
    i = 0

    for img, label, _, _ in test_loader:
        if i < 50:
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

            # print('LV_Dice_%d:' % i, '%.6f' % LV_dice, '||', 'RV_Dice_%d:' % i, '%.6f' % RV_dice, '||'
            #       , 'Myo_Dice_%d:' % i, '%.6f' % Myo_dice)

            i += 1
    print('===============================================')
    LV_Dice_avg = LV_Dice / i
    RV_Dice_avg = RV_Dice / i
    Myo_Dice_avg = Myo_Dice / i
    Mean_metric = (LV_Dice_avg+RV_Dice_avg+Myo_Dice_avg)/3

    return Mean_metric


# if __name__ == '__main__':

    # model = CleanU_Net(1, 4).to(device)
    # model.load_state_dict(torch.load('./3dunet_model_save/weights_200.pth'))
    # test_dataset = MRIdataset.LiverDataset(MRIdataset.test_imagepath, MRIdataset.test_labelpath,
    #                                        MRIdataset.testimg_ids, MRIdataset.testlabel_ids, False)
    #
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    # saveresult2d(test_loader, model)
    # test_model()

    # img = nib.load(r'/home/peng/Desktop/CROP/pred/pred_2.nii.gz').get_data()
    # img = img.flatten()
    # for i in range(len(img)):
    #     if img[i]!=0:
    #         print(img[i])
