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


import matplotlib

matplotlib.use('Agg')
import lasagne
import theano.tensor as T
import numpy as np
import theano
import os
import _pickle as cPickle
# import cPickle
from utils import plotProgress
import time
from utils import soft_dice, hard_dice,weight_soft_dice

# from BatchGenerator import BatchGenerator_2D
from MMs2020.MMS_BatchGenerator import BatchGenerator_2D

# from dataset_utils import load_dataset
from MMs2020.split_labeled import load_dataset

from utils import get_split
import imp
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms import GammaTransform, ConvertSegToOnehotTransform
from batchgenerators.transforms import RandomCropTransform


def create_data_gen_train(patient_data_train, BATCH_SIZE, num_classes,
                          num_workers=2, num_cached_per_worker=2,
                          do_elastic_transform=False, alpha=(0., 1300.), sigma=(10., 13.),
                          do_rotation=False, a_x=(0., 2 * np.pi), a_y=(0., 2 * np.pi), a_z=(0., 2 * np.pi),
                          do_scale=True, scale_range=(0.75, 1.25), seeds=None):
    if seeds is None:
        seeds = [None] * num_workers
    elif seeds == 'range':
        seeds = range(num_workers)
    else:
        assert len(seeds) == num_workers
    data_gen_train = BatchGenerator_2D(patient_data_train, BATCH_SIZE, num_batches=None, seed=False,
                                       PATCH_SIZE=(352, 352))

    tr_transforms = []
    tr_transforms.append(MirrorTransform((0, 1)))
    tr_transforms.append(RndTransform(SpatialTransform((352, 352), list(np.array((352, 352)) // 2),
                                                       do_elastic_transform, alpha,
                                                       sigma,
                                                       do_rotation, a_x, a_y,
                                                       a_z,
                                                       do_scale, scale_range, 'constant', 0, 3, 'constant',
                                                       0, 0,
                                                       random_crop=False), prob=0.67,
                                      alternative_transform=RandomCropTransform((352, 352))))
    tr_transforms.append(ConvertSegToOnehotTransform(range(num_classes), seg_channel=0, output_key='seg_onehot'))

    tr_composed = Compose(tr_transforms)
    tr_mt_gen = MultiThreadedAugmenter(data_gen_train, tr_composed, num_workers, num_cached_per_worker, seeds)
    tr_mt_gen.restart()
    # tr_mt_gen = SingleThreadedAugmenter(data_gen_train, tr_composed)
    return tr_mt_gen


def run(config_file, fold=0):
    cf = imp.load_source('cf', config_file)
    print('fold:', fold)
    dataset_root = cf.dataset_root_mmsB
    print('train path: {}'.format(dataset_root))
    # ==================================================================================================================
    BATCH_SIZE = cf.BATCH_SIZE
    INPUT_PATCH_SIZE = cf.INPUT_PATCH_SIZE
    num_classes = cf.num_classes
    EXPERIMENT_NAME = cf.EXPERIMENT_NAME
    results_dir = os.path.join(cf.results_dir, "fold%d/" % fold)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    n_epochs = cf.n_epochs
    lr_decay = cf.lr_decay
    base_lr = cf.base_lr
    n_batches_per_epoch = cf.n_batches_per_epoch  # 100
    n_test_batches = cf.n_test_batches  # 10
    n_feedbacks_per_epoch = cf.n_feedbacks_per_epoch  # 10
    num_workers = cf.num_workers
    workers_seeds = cf.workers_seeds
    print('basiclr: {},lr-decay: {}'.format(cf.base_lr,cf.lr_decay))
    # ==================================================================================================================

    # this is seeded, will be identical each time
    train_keys, test_keys = get_split(fold)
    print('train_keys:', train_keys)
    print('val_keys:', test_keys)

    train_data = load_dataset(train_keys, root_dir=dataset_root)
    val_data = load_dataset(test_keys, root_dir=dataset_root)

    x_sym = cf.x_sym
    seg_sym = cf.seg_sym

    nt, net, seg_layer = cf.nt_bn, cf.net_bn, cf.seg_layer_bn
    output_layer_for_loss = net
    # draw_to_file(lasagne.layers.get_all_layers(net), os.path.join(results_dir, 'network.png'))

    data_gen_validation = BatchGenerator_2D(val_data, BATCH_SIZE, num_batches=None, seed=False,
                                            PATCH_SIZE=INPUT_PATCH_SIZE)
    # No data augmentation in valuation

    data_gen_validation = MultiThreadedAugmenter(data_gen_validation,
                                                 ConvertSegToOnehotTransform(range(num_classes), 0, "seg_onehot"),
                                                 1, 2, [0])

    # add some weight decay
    l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss,
                                                               lasagne.regularization.l2) * cf.weight_decay

    # the distinction between prediction_train and test is important only if we enable dropout
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False,
                                                 batch_norm_update_averages=False, batch_norm_use_averages=False)

    loss_vec = - weight_soft_dice(prediction_train, seg_sym)

    loss = loss_vec.mean()
    loss += l2_loss
    acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True,
                                                batch_norm_update_averages=False, batch_norm_use_averages=False)
    loss_val = - soft_dice(prediction_test, seg_sym)

    loss_val = loss_val.mean()
    loss_val += l2_loss
    acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    # learning rate has to be a shared variable because we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(base_lr)
    updates = lasagne.updates.adam(T.grad(loss, params), params, learning_rate=learning_rate, beta1=0.9, beta2=0.999)

    dc = hard_dice(prediction_test, seg_sym.argmax(1), num_classes)

    train_fn = theano.function([x_sym, seg_sym], [loss, acc_train, loss_vec], updates=updates)
    val_fn = theano.function([x_sym, seg_sym], [loss_val, acc, dc])

    dice_scores = None
    data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                           num_classes, num_workers=num_workers,
                                           do_elastic_transform=True, alpha=(100., 350.), sigma=(14., 17.),
                                           do_rotation=True, a_x=(0, 2. * np.pi), a_y=(-0.000001, 0.00001),
                                           a_z=(-0.000001, 0.00001), do_scale=True, scale_range=(0.7, 1.3),
                                           seeds=workers_seeds)  # new se has no brain mask

    all_training_losses = []
    all_validation_losses = []
    all_validation_accuracies = []
    all_training_accuracies = []
    all_val_dice_scores = []
    epoch = 0
    val_min = 0

    while epoch < n_epochs:
        if epoch == 100:
            data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                                   num_classes, num_workers=num_workers,
                                                   do_elastic_transform=True, alpha=(0., 250.), sigma=(14., 17.),
                                                   do_rotation=True, a_x=(-2 * np.pi, 2 * np.pi),
                                                   a_y=(-0.000001, 0.00001), a_z=(-0.000001, 0.00001),
                                                   do_scale=True, scale_range=(0.75, 1.25),
                                                   seeds=workers_seeds)  # new se has no brain mask
        if epoch == 125:
            data_gen_train = create_data_gen_train(train_data, BATCH_SIZE,
                                                   num_classes, num_workers=num_workers,
                                                   do_elastic_transform=True, alpha=(0., 150.), sigma=(14., 17.),
                                                   do_rotation=True, a_x=(-2 * np.pi, 2 * np.pi),
                                                   a_y=(-0.000001, 0.00001), a_z=(-0.000001, 0.00001),
                                                   do_scale=True, scale_range=(0.8, 1.2),
                                                   seeds=workers_seeds)  # new se has no brain mask
        epoch_start_time = time.time()
        learning_rate.set_value(np.float32(base_lr * lr_decay ** epoch))
        print("epoch: ", epoch, " learning rate: ", learning_rate.get_value())
        train_loss = 0
        train_acc_tmp = 0
        train_loss_tmp = 0
        batch_ctr = 0
        for data_dict in data_gen_train:
            # first call "__iter__(self)" in class BatchGenerator_2D for iter
            # And then call "__next__()" in class BatchGenerator_2D for looping
            # As a result, it will generate a random batch data every time, much probably different
            data = data_dict["data"].astype(np.float32)
            # print(data.shape) (2,1,352,352) where batchsize = 2
            seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 1).reshape((-1, num_classes))
            if batch_ctr != 0 and batch_ctr % int(np.floor(n_batches_per_epoch / n_feedbacks_per_epoch)) == 0:
                print("number of batches: ", batch_ctr, "/", n_batches_per_epoch)
                print("training_loss since last update: ", \
                      train_loss_tmp / np.floor(n_batches_per_epoch / (n_feedbacks_per_epoch)), " train accuracy: ", \
                      train_acc_tmp / np.floor(n_batches_per_epoch / n_feedbacks_per_epoch))
                """
                n_batches_per_epoch:   How many batches in an epoch, 100 here.
                n_feedbacks_per_epoch: How many feedbacks are given in an epoch, 10 here,it means in an epoch we will 
                                       calculate loss 10 times. 
                
                """
                all_training_losses.append(train_loss_tmp / np.floor(
                    n_batches_per_epoch / (n_feedbacks_per_epoch)))  # for showing and saving result .png
                all_training_accuracies.append(train_acc_tmp / np.floor(n_batches_per_epoch / (n_feedbacks_per_epoch)))
                train_loss_tmp = 0
                train_acc_tmp = 0
                if len(all_val_dice_scores) > 0:
                    dice_scores = np.concatenate(all_val_dice_scores, axis=0).reshape((-1, num_classes))
                plotProgress(all_training_losses, all_training_accuracies, all_validation_losses,
                             all_validation_accuracies, os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME),
                             n_feedbacks_per_epoch, val_dice_scores=dice_scores, dice_labels=["0", "1", "2", "3"])

            if batch_ctr > (n_batches_per_epoch - 1):
                break
            loss_vec, acc, l = train_fn(data, seg)  # type: Array

            loss = loss_vec.mean()

            train_loss += loss
            train_loss_tmp += loss
            train_acc_tmp += acc
            batch_ctr += 1
            # if batch_ctr > (n_batches_per_epoch-1):
            #     break

        train_loss /= n_batches_per_epoch
        print("training loss average on epoch: ", train_loss)

        val_loss = 0
        accuracies = []
        valid_batch_ctr = 0
        all_dice = []
        for data_dict in data_gen_validation:
            data = data_dict["data"].astype(np.float32)  # print(data.shape) (2,1,352,352) where batchsize = 2
            seg = data_dict["seg_onehot"].astype(np.float32).transpose(0, 2, 3, 1).reshape((-1, num_classes))  # (2,4,352,352) --> (2,353,353,4)-->(2*352*352,4)
            w = np.zeros(num_classes, dtype=np.float32)  # [0, 0, 0, 0]
            w[np.unique(seg.argmax(-1))] = 1
            loss, acc, dice = val_fn(data, seg)
            dice[w == 0] = 2
            #  if there are some class was not be classified, abandon it when calculate mean of dice
            all_dice.append(dice)
            val_loss += loss
            accuracies.append(acc)
            valid_batch_ctr += 1
            if valid_batch_ctr > (n_test_batches - 1):
                break
            # Making a valuation every epoch
            # n_test_batches(here is 10) batches in a valuation
        all_dice = np.vstack(all_dice)
        dice_means = np.zeros(num_classes)
        for i in range(num_classes):
            dice_means[i] = all_dice[all_dice[:, i] != 2, i].mean()
        val_loss /= n_test_batches
        print("val loss: ", val_loss)
        print("val acc: ", np.mean(accuracies), "\n")
        print("val dice: ", dice_means)
        print("This epoch took %f sec" % (time.time() - epoch_start_time))
        all_val_dice_scores.append(dice_means)
        all_validation_losses.append(val_loss)
        all_validation_accuracies.append(np.mean(accuracies))
        dice_scores = np.concatenate(all_val_dice_scores, axis=0).reshape((-1, num_classes))
        plotProgress(all_training_losses, all_training_accuracies, all_validation_losses, all_validation_accuracies,
                     os.path.join(results_dir, "%s.png" % EXPERIMENT_NAME), n_feedbacks_per_epoch,
                     val_dice_scores=dice_scores,
                     dice_labels=["0", "1", "2", "3"])
        mul_except_background = np.array([[0],[1],[1],[1]])
        mean_123 = (np.dot(dice_means,mul_except_background))/3
        if mean_123 > val_min:
            print('=========================================================================')
            print('epoch {} val123mean_min change to {:.3f} ,updating saved net params......'.format(epoch,mean_123.item()))
            print('=========================================================================')
            val_min = (np.dot(dice_means,mul_except_background))/3
            with open(os.path.join(results_dir, "%s_Params.pkl" % (EXPERIMENT_NAME)), 'wb') as f:
                cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
        with open(os.path.join(results_dir, "%s_allLossesNAccur.pkl" % (EXPERIMENT_NAME)), 'wb') as f:
            cPickle.dump([all_training_losses, all_training_accuracies, all_validation_losses,
                          all_validation_accuracies, all_val_dice_scores], f)
        epoch += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int, default=0)
    parser.add_argument("-c", help="config file", type=str, default='./UNet2D_config.py')
    args = parser.parse_args()
    run(args.c, args.f)

    """                     ACDC 2D
    BESE_VAL_RESULT:
     ________________________________________________________________
    |  epoch   background         RV           Myo           LV      |
    |  [164] [*0.99925959*   0.82489681    0.88035041    0.93499696 ]|
    |  [241] [ 0.9991104    *0.94312811*   0.89640695    0.95641023 ]|
    |  [266] [ 0.99889517    0.85848427   *0.91053855*   0.95131826 ]|
    |  [212] [ 0.99895227    0.92221969    0.88853139   *0.96168077*]|
    |________________________________________________________________|
    
     _________________________________________________________________
    |LAST_VAL_RESULT:                                                 |
    |   epoch   background         RV           Myo           LV      |                                             |
    |   [299] [ 0.99855882    0.86298871    0.89322245    0.95408523 ]|
    | ________________________________________________________________|  
    
    BEAT MEAN:
        [241] [ 0.9991104     0.94312811    0.89640695    0.95641023 ] [0.93198176]
    
    
    ATTENTION !
        0 : background
        1 : RV
        2 : Myo
        3 : LV
        (from left to right : 0~3)
    """
    """                     MMS 2D lr_decay=0.985
        BESE_VAL_RESULT:
         ________________________________________________________________
        |  epoch   background         LV           Myo           RV      |
        |  [251] [  0.999345     0.79871339   0.80450892     0.83771408] |     
        |  [279] [  0.99876821   0.95691586   0.85037786     0.91591978] |
        |  [128] [  0.99878865   0.94033778   0.88093317     0.81281739] |
      * |  [279] [  0.99876821   0.95691586   0.85037786     0.91591978] |  MEAN : 0.90773783  (BEST)
        |________________________________________________________________|
       
         _________________________________________________________________
        |LAST_VAL_RESULT:                                                 |
        |   epoch   background         LV           Myo           RV      |                                             
        |   [299] [ 0.99794847     0.77378231    0.64043868    0.53501087]|
        |_________________________________________________________________|  
        
                               MMS 2D lr_decay=0.98
         
        best_val dice:  [ 0.99844408 0.95781767 0.84975064 0.91590458 ] epoch[279]
        last_val dice:  [ 0.99758875 0.75564557 0.61036021 0.41069821 ]
        
                               MMS VENDOR-B 2D lr_decay=0.98
         
        best_val dice:  [ 0.99925232 0.95059049 0.88186121 0.93081522 ]  MEAN: [0.92108897] epoch[244]
        last_val dice:  [ 0.99904287 0.92314684 0.83829176 0.91010106 ]
       
       
                               MMS VENDOR-B 2D lr_decay=0.98 INSTANCE_NORM
        best_val dice:  [ 0.99894434 0.92817491 0.86859661 0.89737648 ]  MEAN: [0.89804933] epoch[274]
        last_val dice:  [0.99904323 0.9277119  0.86248142 0.79776305]

        ATTENTION !
            0 : background
            1 : LV
            2 : Myo
            3 : RV
            (from left to right : 0~3)
        """
    # data = np.load('/home/laisong/ACDC2017/mms_vendorAandB_2d_train/pat_005.npy')
    # print(data[1])
    # f = open('/home/laisong/github/Cardiac-segmentation/ACDC2017-master/result/'
    #          'MMS_lasagne/UNet2D_forMMS_VENDOR-B_bn+bigbatch/fold0/UNet2D_forMMS_VENDOR-B_bn+bigbatch_allLossesNAccur.pkl','rb')
    # # # f = open('/home/laisong/github/Cardiac-segmentation/ACDC2017-master/result/ACDC_lasagne/UNet2D_final/fold0/UNet2D_final_allLossesNAccur.pkl','rb')
    # # n = cPickle.load(f)
    # # # #
    # all_training_losses, all_training_accuracies, all_validation_losses,all_validation_accuracies, all_val_dice_scores= cPickle.load(f)
    # # # # # print(np.array(all_val_dice_scores).size)
    # all_val_dice_scores_numpy = np.array(all_val_dice_scores).reshape(300,4)
    # # # print(all_val_dice_scores_numpy[241])
    # # # # print(all_val_dice_scores_numpy[299])
    # # # # [0.99855882 0.86298871 0.89322245 0.95408523]
    # # index = np.argmax(all_val_dice_scores_numpy,axis=0)
    # # # print(index.reshape(4,1))
    # # # for i in range(len(index)):
    # # #     print(all_val_dice_scores_numpy[index[i]])
    # val_dice = all_val_dice_scores_numpy
    # # # val_dice[:,0]= 0
    # # # print(val_dice)
    # mul = np.array([[0],
    #                 [1],
    #                 [1],
    #                 [1]])
    # mean = (np.dot(val_dice,mul)/3)
    # print(mean.argmax(0),mean[mean.argmax(0)])
    # print(all_val_dice_scores[mean.argmax(0).item()])
    # print(all_val_dice_scores[-1])


