import numpy as np
import torch


def dice_coeff(pred, target):
    ims = [pred, target]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.detach().numpy()
        np_ims.append(item)

    pred = np_ims[0]
    target = np_ims[1]

    smooth = 0.0001

    m1 = pred.flatten()  # Flatten
    m2 = target.flatten()  # Flatten

    LV_l1 = 0
    LV_l2 = 0

    RV_l1 = 0
    RV_l2 = 0

    Myo_l1 = 0
    Myo_l2 = 0

    LV_intersection = 0
    RV_intersection = 0
    Myo_intersection = 0
    for i in range(len(m1)):
        if m1[i] == 1:
            LV_l1 += 1
        if (m1[i] == m2[i]) & (m1[i] == 1):
            LV_intersection += 1

        if m1[i] == 3:
            RV_l1 += 1
        if (m1[i] == m2[i]) & (m1[i] == 3):
            RV_intersection += 1

        if m1[i] == 2:
            Myo_l1 += 1
        if (m1[i] == m2[i]) & (m1[i] == 2):
            Myo_intersection += 1
    for i in range(len(m2)):
        if m2[i] == 1:
            LV_l2 += 1
        if m2[i] == 3:
            RV_l2 += 1
        if m2[i] == 2:
            Myo_l2 += 1

    LV_jac = (LV_intersection + smooth) / (-LV_intersection + LV_l1 + LV_l2 + smooth)
    LV_dice = (2. * LV_intersection + smooth) / (LV_l1 + LV_l2 + smooth)

    RV_jac = (RV_intersection + smooth) / (-RV_intersection + RV_l1 + RV_l2 + smooth)
    RV_dice = (2. * RV_intersection + smooth) / (RV_l1 + RV_l2 + smooth)

    Myo_jac = (Myo_intersection + smooth) / (-Myo_intersection + Myo_l1 + Myo_l2 + smooth)
    Myo_dice = (2. * Myo_intersection + smooth) / (Myo_l1 + Myo_l2 + smooth)

    return LV_dice, LV_jac, RV_dice, RV_jac, Myo_dice, Myo_jac


if __name__ == '__main__':
    pred = torch.tensor([[1, 2, 1, 1],
                         [0, 0, 0, 0]])
    target = torch.tensor([[1, 2, 1, 2],
                           [0, 0, 0, 0]])
    print(dice_coeff(pred, target))
