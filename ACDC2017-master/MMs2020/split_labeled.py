import os
from multiprocessing import pool

import SimpleITK as sitk
import xlrd
import numpy as np
import MMs2020.MMs2020_config as MMSCONFIG
from dataset_utils import preprocess_image
import _pickle as cPickle
from skimage.transform import resize

TOTAL_DATA = MMSCONFIG.MMs_TOTAL_DATA
IMG_PATH = MMSCONFIG.IMG_PATH
LABEL_PATH = MMSCONFIG.LABEL_PATH
MMS_2D_TRAIN = MMSCONFIG.MMS_2D_TRAIN
MMS_3D_TRAIN = MMSCONFIG.MMS_3D_TRAIN



MMs_info = xlrd.open_workbook(MMSCONFIG.MMs_info)
sheet = MMs_info.sheet_by_index(0) # 索引的方式，从0开始
# col_data=sheet.col_values(0)  # 获取第一列的内容
# row_data=sheet.row_values(0)  # 获取第一行的内容
External_code = sheet.col_values(0)
Vendor = sheet.col_values(1)
Centre = sheet.col_values(2)
ED = sheet.col_values(3)
ES = sheet.col_values(4)
"""
Attention: it includes title, so you'll take it from index one 
"""
# fname =os.path.join(IMG_PATH, External_code[1]+"_sa.nii.gz")
# print(fname)
# if os.path.isfile(fname):
#     image = sitk.ReadImage(fname)
#     image_spacing = image.GetSpacing()
#     image_as_numpy = sitk.GetArrayFromImage(image).astype(float)
#     print(image_spacing)
# """
#    for example:
#        image's type:<class 'SimpleITK.SimpleITK.Image'>
#        image space:  (1.1824324131011963, 1.1824324131011963, 10.0, 1.0)
#        image_as_numpy's shape:  (30, 10, 320, 320)
#    """
# else:
#     print('open file error!')





def generate_patient_info():
    patient_info={}
    for id in range(1,151):
        patient_info[id] = {}
        patient_info[id]['ed'] = int(ED[id])
        patient_info[id]['es'] = int(ES[id])
        patient_info[id]['code'] = External_code[id]
        patient_info[id]['centre'] = int(Centre[id])
        patient_info[id]['vendor'] = Vendor[id]
    return patient_info

def process_patient(args):
    id, patient_info,folder_out, keep_z_spc = args

    if not os.path.isdir(IMG_PATH):
        return

    images = {}
    fname_img = os.path.join(IMG_PATH,patient_info[id]['code']+'_sa.nii.gz')
    fname_seg = os.path.join(LABEL_PATH,patient_info[id]['code']+'_sa_gt.nii.gz')
    # if os.path.isfile(fname_img):
    images_4d = sitk.ReadImage(fname_img)
    images_4d_spacing = list(images_4d.GetSpacing())
    images_4d_numpy = sitk.GetArrayFromImage(images_4d).astype(float)
    images_spacing = (images_4d_spacing[0],images_4d_spacing[1],images_4d_spacing[2])
    images["ed"] = sitk.GetImageFromArray(images_4d_numpy[patient_info[id]['ed']-1])
    images["ed"].SetSpacing(images_spacing)
    images["es"] = sitk.GetImageFromArray(images_4d_numpy[patient_info[id]['es']-1])
    images["es"].SetSpacing(images_spacing)

    # if os.path.isfile(fname_seg):
    seg_4d = sitk.ReadImage(fname_seg)
    seg_4d_spacing = seg_4d.GetSpacing()
    seg_spacing = (seg_4d_spacing[0],seg_4d_spacing[1],seg_4d_spacing[2])
    seg_4d_numoy = sitk.GetArrayFromImage(seg_4d).astype(float)
    images["ed_seg"] = sitk.GetImageFromArray(seg_4d_numoy[patient_info[id]['ed']-1])
    images["ed_seg"].SetSpacing(seg_spacing)
    images["es_seg"] = sitk.GetImageFromArray(seg_4d_numoy[patient_info[id]['es'] - 1])
    images["es_seg"].SetSpacing(seg_spacing)

    print (id, images["es_seg"].GetSpacing())

    for k in images.keys():
        #print k
        images[k] = preprocess_image(images[k], is_seg=(k == "ed_seg" or k == "es_seg"),
                                     spacing_target=(10, 1.25, 1.25), keep_z_spacing=keep_z_spc)

    img_as_list = []
    for k in ['ed', 'ed_seg', 'es', 'es_seg']:
        if k not in images.keys():
            print (id, "has missing key:", k)
        img_as_list.append(images[k][None])
    try:
        all_img = np.vstack(img_as_list)
    except:
        print (id, "has a problem with spacings")
    np.save(os.path.join(folder_out, "pat_%03.0d" % id), all_img.astype(np.float32))


def run_preprocessing(folder_out=None, keep_z_spacing=True):

    patient_info = generate_patient_info()

    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    with open(os.path.join(folder_out, "patient_info.pkl"), 'wb') as f:
        cPickle.dump(patient_info, f)

    # beware of z spacing!!! see process_patient for more info!
    ids = range(1,151)
    p = pool.Pool(processes=8)
    ZIP = list(zip(ids, [patient_info]*100, [folder_out]*100, [keep_z_spacing]*100))
    p.map(process_patient, zip(ids, [patient_info]*150, [folder_out]*150, [keep_z_spacing]*150))
    p.close()
    p.join()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-out2d", help="folder where to save the data for the 2d network", type=str,default=MMS_2D_TRAIN)
    parser.add_argument("-out3d", help="folder where to save the data for the 3d network", type=str,default=MMS_3D_TRAIN)
    args = parser.parse_args()
    run_preprocessing(args.out2d, True)
    run_preprocessing(args.out3d, False)

