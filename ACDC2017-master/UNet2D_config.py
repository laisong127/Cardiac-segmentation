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
import os
import lasagne
import sys
import theano.tensor as T
from network import build_UNet_relu_BN_ds,build_UNet_relu_INS_ds
from paths import path_mms_2d, results_folder, path_acdc_2d, path_mms_vendorB_2d

# training
x_sym = T.tensor4()
seg_sym = T.matrix()
# dataset_root = path_acdc_2d
dataset_root = path_mms_vendorB_2d
# ======================================================================================================================
np.random.seed(65432)
lasagne.random.set_rng(np.random.RandomState(98765))
sys.setrecursionlimit(2000)
BATCH_SIZE = 6
INPUT_PATCH_SIZE = (352, 352)
num_classes = 4
EXPERIMENT_NAME = "UNet2D_forMMS_VENDOR-B_bn+bigbatch"
# if not os.path.isdir(os.path.join(results_folder, "ACDC_lasagne")):
#     os.mkdir(os.path.join(results_folder, "ACDC_lasagne"))
# results_dir = os.path.join(results_folder, "ACDC_lasagne", EXPERIMENT_NAME)
# if not os.path.isdir(results_dir):
#     os.mkdir(results_dir)
if not os.path.isdir(os.path.join(results_folder, "MMS_lasagne")):
    os.mkdir(os.path.join(results_folder, "MMS_lasagne"))
results_dir = os.path.join(results_folder, "MMS_lasagne", EXPERIMENT_NAME)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
n_epochs = 300
# lr_decay = np.float32(0.985)
lr_decay = np.float32(0.98)
base_lr = np.float32(0.0005)
n_batches_per_epoch = 100
n_test_batches = 50
n_feedbacks_per_epoch = 10.
num_workers = 6
workers_seeds = [123, 1234, 12345, 123456, 1234567, 12345678]
weight_decay = 1e-5
nt_ins, net_ins, seg_layer_ins = build_UNet_relu_INS_ds(1, x_sym, BATCH_SIZE, num_classes, 'same', (None, None), 48, 0.3,
                                           lasagne.nonlinearities.leaky_rectify, bn_axes=(2, 3))
nt_bn, net_bn, seg_layer_bn = build_UNet_relu_BN_ds(1, x_sym, BATCH_SIZE, num_classes, 'same', (None, None), 48, 0.3,
                                           lasagne.nonlinearities.leaky_rectify, bn_axes=(2, 3))

# validation
val_preprocess_fn = None
val_input_img_must_be_divisible_by = 16
val_num_repeats = 2
val_bayesian_prediction = True
val_save_segmentation = True
val_do_mirroring = True
val_plot_segmentation = False
val_min_size = [0, INPUT_PATCH_SIZE[0], INPUT_PATCH_SIZE[1]]

# 4d predictions
dataset_root_raw = '/home/laisong/ACDC2017/training_all'  # ACDC2017 training data path
target_spacing = (None, 1.25, 1.25)
min_size = val_min_size

dataset_root_test = "/home/laisong/ACDC2017/testing/"  # ACDC2017 testing data path
test_out_folder = os.path.join(results_dir, "test_predictions")
if not os.path.isdir(test_out_folder):
    os.mkdir(test_out_folder)
