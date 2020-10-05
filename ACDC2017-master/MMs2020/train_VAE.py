import imp
import os
import time
import _pickle as cPickle

import lasagne
import theano

from MMs2020.MMS_BatchGenerator import BatchGenerator_2D
from MMs2020.split_labeled import load_dataset
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import MirrorTransform, RndTransform, SpatialTransform, RandomCropTransform, \
    ConvertSegToOnehotTransform, Compose
from utils import get_split, F_loss, hard_dice, plotProgress, soft_dice
import theano.tensor as T
import numpy as np
from network import VAE


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
    print('basiclr: {},lr-decay: {}'.format(cf.base_lr, cf.lr_decay))
    # ==================================================================================================================

    # this is seeded, will be identical each time
    train_keys, test_keys = get_split(fold)
    print('train_keys:', train_keys)
    print('val_keys:', test_keys)

    train_data = load_dataset(train_keys, root_dir=dataset_root)
    val_data = load_dataset(test_keys, root_dir=dataset_root)

    x_sym = cf.x_sym
    seg_sym = T.tensor4()

    R_mask = VAE(1, x_sym, BATCH_SIZE, 'same', (None, None), 1, lasagne.nonlinearities.leaky_rectify)
    output_layer_for_loss = R_mask
    # draw_to_file(lasagne.layers.get_all_layers(net), os.path.join(results_dir, 'network.png'))

    data_gen_validation = BatchGenerator_2D(val_data, BATCH_SIZE, num_batches=None, seed=False,
                                            PATCH_SIZE=INPUT_PATCH_SIZE)
    # No data augmentation in valuation

    data_gen_validation = MultiThreadedAugmenter(data_gen_validation,
                                                 ConvertSegToOnehotTransform(range(num_classes), 0, "seg_onehot"),
                                                 1, 2, [0])

    # add some weight decay
    # l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss,
    #                                                            lasagne.regularization.l2) * cf.weight_decay

    # the distinction between prediction_train and test is important only if we enable dropout
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False,
                                                 batch_norm_update_averages=False, batch_norm_use_averages=False)

    loss_vec = F_loss(prediction_train, seg_sym)

    loss = loss_vec.mean()
    # acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True,
                                                batch_norm_update_averages=False, batch_norm_use_averages=False)
    prediction_test = T.round(prediction_test, mode='half_to_even')
    loss_val = F_loss(prediction_test, seg_sym)

    loss_val = loss_val.mean()

    # acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym.argmax(-1)), dtype=theano.config.floatX)

    # learning rate has to be a shared variable because we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(base_lr)
    updates = lasagne.updates.adam(T.grad(loss, params), params, learning_rate=learning_rate, beta1=0.9, beta2=0.999)

    train_fn = theano.function([x_sym, seg_sym], [loss], updates=updates)
    val_fn = theano.function([x_sym, seg_sym], [loss_val])

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
        # learning_rate.set_value(np.float32(base_lr * lr_decay ** epoch))

        print("epoch: ", epoch, " learning rate: ", learning_rate.get_value())
        train_loss = 0

        batch_ctr = 0
        for data_dict in data_gen_train:
            # first call "__iter__(self)" in class BatchGenerator_2D for iter
            # And then call "__next__()" in class BatchGenerator_2D for looping
            # As a result, it will generate a random batch data every time, much probably different

            seg = data_dict["seg_onehot"].astype(np.float32)
            seg = np.argmax(seg,1)
            seg = seg[:,np.newaxis,...].astype(np.float32)

            if batch_ctr > (n_batches_per_epoch - 1):
                break
            loss = train_fn(seg, seg)  # type:numpy.narray
            # print('batch loss:',loss[0])
            train_loss += loss[0].item()
            batch_ctr += 1

        train_loss /= n_batches_per_epoch
        print("training loss average on epoch: ", train_loss)

        val_loss = 0
        valid_batch_ctr = 0

        for data_dict in data_gen_validation:
            seg = data_dict["seg_onehot"].astype(np.float32)
            seg = np.argmax(seg, 1)
            seg = seg[:, np.newaxis, ...].astype(np.float32)
            loss = val_fn(seg, seg)
            val_loss += loss[0].item()

            valid_batch_ctr += 1
            if valid_batch_ctr > (n_test_batches - 1):
                break
        val_loss /= n_test_batches
        print('val_loss:',val_loss)
        with open(os.path.join(results_dir, "%s_Params.pkl" % (EXPERIMENT_NAME)), 'wb') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)
            # Making a valuation every epoch
            # n_test_batches(here is 10) batches in a valuation

        epoch += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="fold", type=int, default=0)
    parser.add_argument("-c", help="config file", type=str, default='../UNet2D_config.py')
    args = parser.parse_args()
    run(args.c, args.f)