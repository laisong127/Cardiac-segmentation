# import numpy as np
# pred   = [[0.9,0.2],
#            [0.1,0.8],
#            [0,0],
#            [0,0]]
# numpy = np.array(pred)
# # data = np.zeros(tuple([1] + [1] + list(new_shp[1:])), dtype=np.float32)
# # print(np.array(pred).mean(0))
# print(numpy[None])
import lasagne
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
from skimage.morphology import label
x_sym = T.matrix()
from lasagne.layers import InputLayer, DenseLayer
l_in = InputLayer((8, 20))
l1 = DenseLayer(l_in, num_units=2)
inputs = np.random.randn(8,20).astype(np.float32)
out = lasagne.layers.get_output(l1, x_sym)
pred = theano.function([x_sym],out)
outputs = pred(inputs)
print(outputs.shape)
x=np.array([[1,2,3],[1,2,1]])
y = np.array([[1,1,3],[1,2,1]])
z=x-y
z = z**2
# for i,tpe in enumerate(['ed', 'es']):
#     print(i,tpe)
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
def weight_soft_dice(y_pred, y_true, w=None):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    if w is None:
        w = [1, 1, 4, 2]
    y_pred_i = []
    count = np.sum(np.array(w))
    for i in range(4):
        y_pred_i.append(y_pred[:,i])
    y_true_i = []
    for i in range(4):
        y_true_i.append(y_true[:,i])
    intersect = []
    denominator = []
    dice_scores_i = []
    for i in range(4):
        intersect.append(T.sum(y_pred_i[i] * y_true_i[i], 0))
        denominator.append(T.sum(y_pred_i[i], 0) + T.sum(y_true_i[i], 0))
        dice_scores_i.append(T.constant(2) * intersect[i] / (denominator[i] + T.constant(1e-6)))
    dice_scores = (T.constant(w[0])*dice_scores_i[0]+T.constant(w[1])*dice_scores_i[1]\
                  +T.constant(w[2])*dice_scores_i[2]+T.constant(w[3])*dice_scores_i[3])
    x = T.matrix('total')
    z = x/8
    divide = theano.function([x],z)
    dice_scores = divide(dice_scores)

    return dice_scores
x = np.array([[[0,0,0],
              [0,0,0],
              [0,0,0]],
              [[1, 0, 1],
               [2, 0, 0],
               [3, 1, 0]]
              ])
mask = np.array(x != 0).astype(int)
lbls = label(mask,4 )
lbls_sizes = [np.sum(lbls==i) for i in np.unique(lbls)]
largest_region = np.argmax(lbls_sizes[1:]) + 1 # from 1 because need excluding the background
x[lbls != largest_region]=0
print(x)
print(np.argmax(x))
print(x)
print(label(x,4))

# data = np.array([1,2,3])
# print(data)
# data = np.vstack([data]*3)
#
#
# w = np.zeros(4, dtype=np.float32)
# print(w)
# w[[2,3]] = 1
# print(w)
# print([w==0])
# dice = np.array([0.1,0.2,0.3,0.4])
# print(dice.shape)
# import numpy as np
# import theano.tensor as T
# from theano import function
#
#
# x=T.dvector()
# # y=T.matrix('y')
# z=x
# a=T.vector()
# out=a
# f=function([a],out,allow_input_downcast=True)
#
# print (f([0.1,0.2]))
#
# f=function([x],z)
# dice = f(dice)
# dice[w==0] = 2
# print(dice)
# print(dice[1] != 2)
#
# dice = np.array(dice)
# # dice[w == 0] = 2
# print(dice.argmax(-1))

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