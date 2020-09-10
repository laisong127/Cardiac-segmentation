# import numpy as np
# pred   = [[0.9,0.2],
#            [0.1,0.8],
#            [0,0],
#            [0,0]]
# numpy = np.array(pred)
# # data = np.zeros(tuple([1] + [1] + list(new_shp[1:])), dtype=np.float32)
# # print(np.array(pred).mean(0))
# print(numpy[None])
import matplotlib
import theano
from skimage.transform import resize

from UNet2D_config import dataset_root_raw

# matplotlib.use('agg')
import numpy as np
import _pickle as cPickle
# import lasagne
# import theano as T
import os
import sys
sys.path.append("../")
import theano.tensor as T
import SimpleITK as sitk
from utils import predict_patient_2D_net, get_split, softmax_helper, resize_softmax_output
import imp
from test_set.preprocess_test_set import generate_patient_info, preprocess
from dataset_utils import resize_image
import numpy as np
from PIL import Image

from skimage import transform, data
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

w = np.zeros(4, dtype=np.float32)
print(w)
w[[2,3]] = 1
print(w)
print([w==0])
dice = np.array([0.1,0.2,0.3,0.4])
print(dice.shape)
import numpy as np
import theano.tensor as T
from theano import function

x=T.dvector()
# y=T.matrix('y')
z=x
a=T.vector()
out=a
f=function([a],out,allow_input_downcast=True)

print (f([0.1,0.2]))

f=function([x],z)
dice = f(dice)
dice[w==0] = 2
print(dice)
print(dice[1] != 2)

dice = np.array(dice)
# dice[w == 0] = 2
print(dice.argmax(-1))

#=====================================================================
# im = Image.open('/home/laisong/cvlab/trainCvlab/img/train001.png')
#
# img = data.camera()
# print(np.array(img).shape)
# print(np.array(img))
# dst = transform.resize(img, (256, 256))*255
# print(np.array(dst).shape)
# print(np.array(dst))
# plt.figure('resize')
#
# plt.subplot(131)
# plt.title('before resize')
# plt.imshow(img, plt.cm.gray)
#
# plt.subplot(132)
# plt.title('resize')
# plt.imshow(dst, plt.cm.gray)
#
# plt.subplot(133)
# plt.title('001')
# plt.imshow(im)
# plt.tight_layout(h_pad=2.0)
#
# plt.show()
#===========================================================================

#===========================================================================
# im = Image.open('/home/laisong/cvlab/trainCvlab/img/train001.png')
# im_as_numpy = np.array(im)
# im_shape = np.array(im_as_numpy.shape)
# new_img = resize(im_as_numpy,im_shape//2, order=3, mode='edge')
# new_im = Image.fromarray((new_img)).convert('L')
#
# new_im.save('test.png')
# new_im.show()
#===========================================================================

#===========================================================================
# ed_image = sitk.ReadImage('/home/laisong/ACDC2017/training_all/patient001/patient001_frame01.nii.gz')
# old_spacing = np.array(ed_image.GetSpacing())
# print('old_spacing: {}'.format(old_spacing))
# old_img = sitk.GetArrayFromImage(ed_image).astype(float)
# old_shape = sitk.GetArrayFromImage(ed_image).astype(float).shape
# print('old_shape: {}'.format(old_shape))
# new_spacing = 2*old_spacing
# print(new_spacing)
# respacing_img = resize_image(sitk.GetArrayFromImage(ed_image).astype(float), old_spacing, new_spacing, 3).astype(
#             np.float32)
# print('respacing_img_shape: {}'.format(respacing_img.shape))
# print(old_img==respacing_img)
#===========================================================================