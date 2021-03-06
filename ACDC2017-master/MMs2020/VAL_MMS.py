import time

import lasagne
import theano
import _pickle as cPickle
from MMs2020.MMS_BatchGenerator import BatchGenerator_2D
from MMs2020.split_labeled import load_dataset
from dataset_utils import load_dataset as ACDC_load_dataset
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import ConvertSegToOnehotTransform
import UNet2D_config as cf
import numpy as np
from utils import soft_dice, hard_dice, get_split
# Params_B = '/home/laisong/github/Cardiac-segmentation/ACDC2017-master/result/' \
#            'MMS_lasagne/UNet2D_forMMS_VENDOR-B_final/fold0/UNet2D_forMMS_VENDOR-B_final_Params.pkl'
Params_B = '/home/laisong/github/Cardiac-segmentation/ACDC2017-master/result/' \
           'MMS_lasagne/UNet2D_forMMS_VENDOR-B_bn+bigbatch/fold0/UNet2D_forMMS_VENDOR-B_bn+bigbatch_Params.pkl'
Params_A = '/home/laisong/github/Cardiac-segmentation/ACDC2017-master/result/' \
           'MMS_lasagne/UNet2D_forMMS_final/fold0/UNet2D_forMMS_final_Params.pkl'
train_keys, test_keys = get_split(0)
all_keys = range(1,151)
print(test_keys)
dataset_root_B = '/home/laisong/ACDC2017/mms_vendorB_2d_train'
dataset_root_A = '/home/laisong/ACDC2017/mms_vendorA_2d_train'
dataset_root_ACDC = '/home/laisong/ACDC2017/2d_train'
BATCH_SIZE = cf.BATCH_SIZE
num_classes = cf.num_classes
x_sym = cf.x_sym
seg_sym = cf.seg_sym
nt, net, seg_layer = cf.nt_bn, cf.net_bn, cf.seg_layer_bn
output_layer_for_loss = net
output_layer = seg_layer


with open(Params_B, 'rb') as f:
    params = cPickle.load(f)
    lasagne.layers.set_all_param_values(output_layer, params)  # load net's params

val_data = load_dataset(test_keys, root_dir=dataset_root_A)
data_gen_validation = BatchGenerator_2D(val_data, BATCH_SIZE, num_batches=None, seed=False, PATCH_SIZE=cf.INPUT_PATCH_SIZE)
data_gen_validation = MultiThreadedAugmenter(data_gen_validation,
                                             ConvertSegToOnehotTransform(range(num_classes), 0, "seg_onehot"),
                                             1, 2, [0])
val_loss = 0
all_training_losses = []
all_validation_losses = []
all_validation_accuracies = []
all_training_accuracies = []
all_val_dice_scores = []
epoch = 0
val_min = 0
accuracies = []
valid_batch_ctr = 0
all_dice = []
prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True,
                                                batch_norm_update_averages=False, batch_norm_use_averages=False)
dc = hard_dice(prediction_test, seg_sym.argmax(1), num_classes)
val_fn = theano.function([x_sym, seg_sym], dc)

for data_dict in data_gen_validation:
    data = data_dict["data"].astype(np.float32)  # print(data.shape) (2,1,352,352) where batchsize = 2
    seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 1).reshape((-1, num_classes))  # (2,4,352,352) --> (2,353,353,4)-->(2*352*352,4)
    w = np.zeros(num_classes, dtype=np.float32)  # [0, 0, 0, 0]
    w[np.unique(seg.argmax(-1))] = 1
    dice = val_fn(data, seg)
    dice[w == 0] = 2
    #  if there are some class was not be classified, abandon it when calculate mean of dice
    all_dice.append(dice)
    valid_batch_ctr += 1
    if valid_batch_ctr > (150 - 1):
        break
    # Making a valuation every epoch
    # n_test_batches(here is 10) batches in a valuation
all_dice = np.vstack(all_dice)
dice_means = np.zeros(num_classes)
for i in range(num_classes):
    dice_means[i] = all_dice[all_dice[:, i] != 2, i].mean()

print("val dice: ", dice_means)

"""
TRAIN_B---> B-test.keys()   val dice:  [ 0.99899477 0.90863502 0.82922918 0.85747451 ] batchsize = 2 
TRAIN_B---> B-train.keys()  val dice:  [ 0.9987281  0.9143998  0.85980201 0.85571545 ]

TRAIN_B---> A             **val dice:  [ 0.99295223 0.39576975 0.28044626 0.2094229  ]** batchsize=2 test 750 batch
TRAIN_B---> A  batchsize=4  val dice:  [ 0.99453157 0.62202203 0.43553695 0.30885321 ]

TRAIN_A---> A-train.keys()  val dice:  [ 0.9986589  0.9201231  0.82787943 0.81763309 ]
TRAIN_A---> A-test.keys()   val dice:  [ 0.99822176 0.89156944 0.81024128 0.78973645 ] batchsize=2
TRAIN-A---> B               val dice:  [ 0.99797779 0.82399106 0.71177512 0.7709527  ] batchsize=2 test 750 batch
"""

