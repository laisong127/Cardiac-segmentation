# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import _pickle as cPickle
import SimpleITK as sitk
import theano

from test_set.preprocess_test_set import generate_patient_info
import os
import sys
sys.path.append("../")
from utils import postprocess_prediction, get_split, soft_dice, hard_dice, test_hard_dice
from skimage.transform import resize
import imp
CONFIG_FILE_2D = '/home/laisong/github/Cardiac-segmentation/ACDC2017-master/UNet2D_config.py'
import paths
import MMs2020.MMs2020_config as MMSCONFIG
import theano.tensor as T


def test_dice(pred_path,true_path,file_index,vendor_path):
    f = open(os.path.join(vendor_path,'patient_info.pkl'), 'rb')
    patient_info = cPickle.load(f)  # 读出文件的数据个数
    Dice = []
    print(file_index)

    for i in file_index:
        for tpe in ['ED','ES']:
            data_path = os.path.join(pred_path,'patient%03d'%i+'_%s'%tpe+'.nii.gz')
            pred_data = sitk.ReadImage(data_path)
            pred_data = sitk.GetArrayFromImage(pred_data)

            label_path = os.path.join(true_path,patient_info[i]['code']+'_%s'%(tpe.lower()+'_seg.gt.nii.gz'))
            label_data = sitk.ReadImage(label_path)
            label_data = sitk.GetArrayFromImage(label_data)

            dice_tmp = test_hard_dice(pred_data,label_data,n_classes=4)
            Dice.append(dice_tmp)
    Dice = np.array(Dice)
    Dice_mean = np.mean(Dice,axis=0)
    return Dice_mean

def run(config_file_2d, output_folder):
    cf_2d = imp.load_source("cf_2d", config_file_2d)

    # cf_3d = imp.load_source("cf_3d", config_file_3d)

    dataset_base_dir = cf_2d.dataset_root_test

    # results_folder_3D = os.path.join(cf_3d.results_dir, "test_predictions/")

    results_folder_2D = cf_2d.test_out_folder
    f = open('/home/laisong/ACDC2017/mms_vendorB_2d_train/patient_info.pkl', 'rb')
    patient_info = cPickle.load(f)  # 读出文件的数据个数

    if not os.path.isdir(output_folder):  # if folder not exist, create it
        os.mkdir(output_folder)


    def resize_softmax_pred(softmax_output, new_shape, order=3):
        reshaped = np.zeros([len(softmax_output)] + list(new_shape), dtype=float)
        for i in range(len(softmax_output)):
            reshaped[i] = resize(softmax_output[i].astype(float), new_shape, order, mode="constant", cval=0, clip=True)
        return reshaped

    train_keys, test_keys = get_split(0)
    print(test_keys)
    for patient in test_keys:

        ed = 0
        es = 2

        for tpe in ['ed', 'es']:
            raw_itk = sitk.ReadImage(os.path.join(MMSCONFIG.MMs_TOTAL_DATA,'Img', patient_info[patient]['code']+'_sa.nii.gz'))
            raw = sitk.GetArrayFromImage(raw_itk)
            print('raw.shape:',raw.shape[1:])
            for f in range(1):  # should be 5 folder totally
                # res_3d = np.load(os.path.join(results_folder_3D, "fold%d" % f, "patient%03.0d_3D_net.npz" % patient))
                res_2d = np.load(os.path.join(results_folder_2D,'fold0','patient%03d_2D_net.npz'%patient))
                print(res_2d[tpe].shape)
                # resize softmax to original image size
                # softmax_3d = resize_softmax_pred(res_3d[tpe], raw.shape, 3)
                softmax_2d = resize_softmax_pred(res_2d[tpe], raw.shape[1:], 3)  # softmax_2d shape(4,10,256,232)
                print(softmax_2d.shape)
                """
                res_2d[tpe].shape may be changed because of the transfering of spacing,
                this operation aims to recover the res_2d[tpe] to original shape(raw.shape)
                """

                # all_softmax += [softmax_3d[None], softmax_2d[None]]
            # predicted_seg = postprocess_prediction(np.vstack(all_softmax).mean(0).argmax(0))
            predicted_seg = postprocess_prediction(softmax_2d.argmax(0))

            itk_seg = sitk.GetImageFromArray(predicted_seg.astype(np.uint8))
            # itk_seg.CopyInformation(raw_itk)

            sitk.WriteImage(itk_seg, os.path.join(output_folder, "patient%03.0d_%s.nii.gz" % (patient, tpe.upper())))



if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-c2d", help="config file for 2d network", type=str, default=CONFIG_FILE_2D)
    # # parser.add_argument("-c3d", help="config file for 3d network", type=str)
    # parser.add_argument("-o", help="output folder", type=str, default='./submit_MMS_file')
    # args = parser.parse_args()
    # run(args.c2d, args.o)
    pred_path = '/home/laisong/github/submit_MMS_file'
    label_path = '/home/laisong/MMs2020/MMS_ED_ES/label'
    train_keys,test_keys = get_split(0)
    vendor_path = paths.path_mms_vendorAandB_2d
    dice = test_dice(pred_path,label_path,test_keys,vendor_path)
    print(dice)