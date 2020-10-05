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



from lasagne.layers import InputLayer, DimshuffleLayer, ReshapeLayer, ConcatLayer, NonlinearityLayer, instance_norm, \
    ElemwiseSumLayer, DropoutLayer, Pool2DLayer, Upscale2DLayer, instance_norm, batch_norm

from collections import OrderedDict
from lasagne.init import HeNormal
from lasagne.nonlinearities import linear
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import TransposedConv2DLayer,DenseLayer,NINLayer




def build_UNet_relu_INS_ds(n_input_channels=1, input_var=None, BATCH_SIZE=None, num_output_classes=2, pad='same',
                          input_dim=(128, 128), base_n_filters=64, dropout=None,
                          nonlinearity=lasagne.nonlinearities.rectify, bn_axes=(2, 3)):
    # nonlinearity=lasagne.nonlinearities.rectify ---> ReLu(x)
    net = OrderedDict()  # convert net to a dict such as [('input',InputLayer((BATCH_SIZE, n_input_channels, input_dim[0],
                         # input_dim[1]), input_var)),.....]
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]), input_var)

    net['contr_1_1'] = instance_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['contr_1_2'] = instance_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)
    #  Double Conv ---> ReLu

    net['contr_2_1'] = instance_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['contr_2_2'] = instance_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    l = net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['contr_3_1'] = instance_norm(ConvLayer(l, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['contr_3_2'] = instance_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    l = net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['contr_4_1'] = instance_norm(ConvLayer(l, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['contr_4_2'] = instance_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['encode_1'] = instance_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad,
                                           W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    l = net['encode_2'] = instance_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity,
                                               pad=pad, W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['upscale1'] = Upscale2DLayer(l, 2)

    l = net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_1_1'] = instance_norm(ConvLayer(l, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    l = net['expand_1_2'] = instance_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity,
                                                 pad=pad, W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['upscale2'] = Upscale2DLayer(l, 2)

    l = net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_2_1'] = instance_norm(ConvLayer(l, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    ds2 = l = net['expand_2_2'] = instance_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3,
                                                       nonlinearity=nonlinearity, pad=pad,
                                                       W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['upscale3'] = Upscale2DLayer(l, 2)

    l = net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_3_1'] = instance_norm(ConvLayer(l, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    l = net['expand_3_2'] = instance_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity,
                                                 pad=pad, W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['upscale4'] = Upscale2DLayer(l, 2)

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = instance_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)
    net['expand_4_2'] = instance_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")))  #, axes=bn_axes)

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)

    ds2_1x1_conv = ConvLayer(ds2, num_output_classes, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale = Upscale2DLayer(ds2_1x1_conv, 2)
    ds3_1x1_conv = ConvLayer(net['expand_3_2'], num_output_classes, 1, 1, 'same', nonlinearity=linear,
                             W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale_ds3_sum = ElemwiseSumLayer((ds1_ds2_sum_upscale, ds3_1x1_conv))
    ds1_ds2_sum_upscale_ds3_sum_upscale = Upscale2DLayer(ds1_ds2_sum_upscale_ds3_sum, 2)

    l = seg_layer = ElemwiseSumLayer((net['output_segmentation'], ds1_ds2_sum_upscale_ds3_sum_upscale))

    net['dimshuffle'] = DimshuffleLayer(l, (0, 2, 3, 1))  # change dimensions
    batch_size, n_rows, n_cols, _ = lasagne.layers.get_output(net['dimshuffle']).shape
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (batch_size * n_rows * n_cols, num_output_classes))  # reshape tensor as [-1,4]
    net['output_flattened'] = NonlinearityLayer(net['reshapeSeg'], nonlinearity=lasagne.nonlinearities.softmax)
    #  softmax faltten tensor
    #  This activation function gets applied row-wise.

    return net, net['output_flattened'], seg_layer


def build_UNet_relu_BN_ds(n_input_channels=1, input_var=None, BATCH_SIZE=None, num_output_classes=2, pad='same',
                          input_dim=(128, 128), base_n_filters=64, dropout=None,
                          nonlinearity=lasagne.nonlinearities.rectify, bn_axes=(2, 3)):
    # nonlinearity=lasagne.nonlinearities.rectify ---> ReLu(x)
    net = OrderedDict()  # convert net to a dict such as [('input',InputLayer((BATCH_SIZE, n_input_channels, input_dim[0],
                         # input_dim[1]), input_var)),.....]
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]), input_var)

    net['contr_1_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_1_2'] = batch_norm(ConvLayer(net['contr_1_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['pool1'] = Pool2DLayer(net['contr_1_2'], 2)
    #  Double Conv ---> ReLu

    net['contr_2_1'] = batch_norm(ConvLayer(net['pool1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_2_2'] = batch_norm(ConvLayer(net['contr_2_1'], base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['pool2'] = Pool2DLayer(net['contr_2_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['contr_3_1'] = batch_norm(ConvLayer(l, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_3_2'] = batch_norm(ConvLayer(net['contr_3_1'], base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['pool3'] = Pool2DLayer(net['contr_3_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['contr_4_1'] = batch_norm(ConvLayer(l, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['contr_4_2'] = batch_norm(ConvLayer(net['contr_4_1'], base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['pool4'] = Pool2DLayer(net['contr_4_2'], 2)
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)

    net['encode_1'] = batch_norm(ConvLayer(l, base_n_filters*16, 3, nonlinearity=nonlinearity, pad=pad,
                                           W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['encode_2'] = batch_norm(ConvLayer(net['encode_1'], base_n_filters*16, 3, nonlinearity=nonlinearity,
                                               pad=pad, W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale1'] = Upscale2DLayer(l, 2)

    l = net['concat1'] = ConcatLayer([net['upscale1'], net['contr_4_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_1_1'] = batch_norm(ConvLayer(l, base_n_filters*8, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['expand_1_2'] = batch_norm(ConvLayer(net['expand_1_1'], base_n_filters*8, 3, nonlinearity=nonlinearity,
                                                 pad=pad, W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale2'] = Upscale2DLayer(l, 2)

    l = net['concat2'] = ConcatLayer([net['upscale2'], net['contr_3_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_2_1'] = batch_norm(ConvLayer(l, base_n_filters*4, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    ds2 = l = net['expand_2_2'] = batch_norm(ConvLayer(net['expand_2_1'], base_n_filters*4, 3,
                                                       nonlinearity=nonlinearity, pad=pad,
                                                       W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale3'] = Upscale2DLayer(l, 2)

    l = net['concat3'] = ConcatLayer([net['upscale3'], net['contr_2_2']], cropping=(None, None, "center", "center"))
    if dropout is not None:
        l = DropoutLayer(l, p=dropout)
    net['expand_3_1'] = batch_norm(ConvLayer(l, base_n_filters*2, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    l = net['expand_3_2'] = batch_norm(ConvLayer(net['expand_3_1'], base_n_filters*2, 3, nonlinearity=nonlinearity,
                                                 pad=pad, W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['upscale4'] = Upscale2DLayer(l, 2)

    net['concat4'] = ConcatLayer([net['upscale4'], net['contr_1_2']], cropping=(None, None, "center", "center"))
    net['expand_4_1'] = batch_norm(ConvLayer(net['concat4'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)
    net['expand_4_2'] = batch_norm(ConvLayer(net['expand_4_1'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                             W=lasagne.init.HeNormal(gain="relu")), axes=bn_axes)

    net['output_segmentation'] = ConvLayer(net['expand_4_2'], num_output_classes, 1, nonlinearity=None)

    ds2_1x1_conv = ConvLayer(ds2, num_output_classes, 1, 1, 'same', nonlinearity=linear, W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale = Upscale2DLayer(ds2_1x1_conv, 2)
    ds3_1x1_conv = ConvLayer(net['expand_3_2'], num_output_classes, 1, 1, 'same', nonlinearity=linear,
                             W=HeNormal(gain='relu'))
    ds1_ds2_sum_upscale_ds3_sum = ElemwiseSumLayer((ds1_ds2_sum_upscale, ds3_1x1_conv))
    ds1_ds2_sum_upscale_ds3_sum_upscale = Upscale2DLayer(ds1_ds2_sum_upscale_ds3_sum, 2)

    l = seg_layer = ElemwiseSumLayer((net['output_segmentation'], ds1_ds2_sum_upscale_ds3_sum_upscale))

    net['dimshuffle'] = DimshuffleLayer(l, (0, 2, 3, 1))  # change dimensions
    batch_size, n_rows, n_cols, _ = lasagne.layers.get_output(net['dimshuffle']).shape
    net['reshapeSeg'] = ReshapeLayer(net['dimshuffle'], (batch_size * n_rows * n_cols, num_output_classes))  # reshape tensor as [-1,4]
    net['output_flattened'] = NonlinearityLayer(net['reshapeSeg'], nonlinearity=lasagne.nonlinearities.softmax) # softmax faltten tensor

    return net, net['output_flattened'], seg_layer


def VAE(n_input_channels=1, input_var=None, BATCH_SIZE=None, pad='same',
                          input_dim=(128, 128), base_n_filters=64,
                          nonlinearity=lasagne.nonlinearities.rectify):
    net = OrderedDict()
    net['input'] = InputLayer((BATCH_SIZE, n_input_channels, input_dim[0], input_dim[1]), input_var)
    net['contr_1'] = batch_norm(ConvLayer(net['input'], base_n_filters, 3, nonlinearity=nonlinearity, pad=pad,
                                              W=lasagne.init.HeNormal(gain="relu")))
    net['contr__2_1'] = batch_norm(ConvLayer(net['contr_1'], base_n_filters, 3,stride=(2,2), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['contr__2_2'] = batch_norm(ConvLayer(net['contr__2_1'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))

    net['contr__3_1'] = batch_norm(ConvLayer(net['contr__2_2'], base_n_filters, 3, stride=(2, 2), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['contr__3_2'] = batch_norm(ConvLayer(net['contr__3_1'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))

    net['contr__4_1'] = batch_norm(ConvLayer(net['contr__3_2'], base_n_filters, 3, stride=(2, 2), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['contr__4_2'] = batch_norm(ConvLayer(net['contr__4_1'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))

    net['contr__5_1'] = batch_norm(ConvLayer(net['contr__4_2'], base_n_filters, 3, stride=(2, 2), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['contr__5_2'] = batch_norm(ConvLayer(net['contr__5_1'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['contr__6_1'] = batch_norm(ConvLayer(net['contr__5_2'], base_n_filters, 3, stride=(2, 2), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['contr__6_2'] = batch_norm(ConvLayer(net['contr__6_1'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['dimshuffle'] = DimshuffleLayer(net['contr__6_2'], (0, 2, 3, 1))  # change dimensions
    batch_size, n_rows, n_cols, _ = lasagne.layers.get_output(net['dimshuffle']).shape
    net['flatten'] = ReshapeLayer(net['dimshuffle'],(BATCH_SIZE,121))  # reshape tensor as [-1,1]

    net['fc_1'] = DenseLayer(net['flatten'],num_units=64,nonlinearity=nonlinearity)
    net['fc_2']=DenseLayer(net['fc_1'],121,nonlinearity=nonlinearity)
    net['flattened_2d'] = ReshapeLayer(net['fc_2'],(batch_size,n_input_channels,n_rows,n_cols))

    net['decode__1_1'] = batch_norm(ConvLayer(net['flattened_2d'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['decode__1_2'] = batch_norm(TransposedConv2DLayer(net['decode__1_1'], base_n_filters, 4, stride=(2, 2), nonlinearity=nonlinearity,crop=1,
                                            W=lasagne.init.HeNormal(gain="relu")))

    net['decode__2_1'] = batch_norm(ConvLayer(net['decode__1_2'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                                            W=lasagne.init.HeNormal(gain="relu")))
    net['decode__2_2'] = batch_norm(TransposedConv2DLayer(net['decode__2_1'], base_n_filters, 4, stride=(2, 2), nonlinearity=nonlinearity,crop=1,
                                            W=lasagne.init.HeNormal(gain="relu")))

    net['decode__3_1'] = batch_norm(
        ConvLayer(net['decode__2_2'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                  W=lasagne.init.HeNormal(gain="relu")))
    net['decode__3_2'] = batch_norm(
        TransposedConv2DLayer(net['decode__3_1'], base_n_filters, 4, stride=(2, 2), nonlinearity=nonlinearity,crop=1,
                              W=lasagne.init.HeNormal(gain="relu")))

    net['decode__4_1'] = batch_norm(
        ConvLayer(net['decode__3_2'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity, pad=pad,
                  W=lasagne.init.HeNormal(gain="relu")))
    net['decode__4_2'] = batch_norm(
        TransposedConv2DLayer(net['decode__4_1'], base_n_filters, 4, stride=(2, 2), nonlinearity=nonlinearity,crop=1,
                              W=lasagne.init.HeNormal(gain="relu")))

    net['decode__5_1'] = batch_norm(
        ConvLayer(net['decode__4_2'], base_n_filters, 3, stride=(1, 1), nonlinearity=nonlinearity,pad=pad,
                  W=lasagne.init.HeNormal(gain="relu")))
    net['decode__5_2'] = batch_norm(
        TransposedConv2DLayer(net['decode__5_1'], base_n_filters, 4, stride=(2, 2), nonlinearity=nonlinearity,crop=1,
                              W=lasagne.init.HeNormal(gain="relu")))
    return  net['decode__5_2']
    pass

if __name__=='__main__':
    import theano.tensor as T
    import theano
    import numpy as np

    x_sym = T.tensor4()
    # nt_ins, net_ins, seg_layer_ins = build_UNet_relu_BN_ds(1, x_sym, 2, 4, 'same', (None, None), 48,
    #                                                         0.3,lasagne.nonlinearities.leaky_rectify, bn_axes=(2, 3))
    nt_vae = VAE(1, x_sym, 2, 'same', (None, None), 1,lasagne.nonlinearities.leaky_rectify)
    prediction_train = lasagne.layers.get_output(nt_vae, x_sym, deterministic=False,
                                                 batch_norm_update_averages=False, batch_norm_use_averages=False)
    pred = theano.function([x_sym],prediction_train)
    inputs = np.random.randn(2,1,352,352).astype(np.float32)
    print(inputs.shape)
    _ = pred(inputs)
    print(type(_),_.shape)


