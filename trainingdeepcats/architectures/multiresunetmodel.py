"""
The MultiresUnet Model
Author: Gerald M

Adapted from paper [MultiResUNet : Rethinking the U-Net Architecture for
Multimodal Biomedical Image Segmentation](10.1016/j.neunet.2019.08.025)
"""

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate, Conv2D, Dropout, UpSampling2D, MaxPool2D, Add, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD

import losses

# Multi-resolution Inception style filters
def _multiresblock(inputs, filter_size, actfn):
    """
    Multi-resolution block in the style of Inception module. This concatenates
    the features from different convolution filters as an approximation for a
    3x3, 5x5 and 7x7 filter size convolution.
    """

    alpha = 1.67
    filter_size1 = int(alpha*filter_size*0.167)
    filter_size2 = int(alpha*filter_size*0.333)
    filter_size3 = int(alpha*filter_size*0.5)
    filter_size4 = int(filter_size1+filter_size2+filter_size3)

    cnn1 = Conv2D(filter_size1, (3,3), padding='same', activation=actfn, kernel_initializer = 'he_normal')(inputs)
    cnn2 = Conv2D(filter_size2, (3,3), padding='same', activation=actfn, kernel_initializer = 'he_normal')(cnn1)
    cnn3 = Conv2D(filter_size3, (3,3), padding='same', activation=actfn, kernel_initializer = 'he_normal')(cnn2)

    cnn = Conv2D(filter_size4, (1,1), padding='same', activation=actfn, kernel_initializer = 'he_normal')(inputs)

    concat = Concatenate()([cnn1, cnn2, cnn3])
    concat = BatchNormalization()(concat)

    add = Add()([concat, cnn])
    add = Activation(act)(add)
    add = BatchNormalization()(add)

    return add

# Concatenation path from encoder to decoder. This performs convolutions on the
# encoder segment to match higher level feature set seen in decoder segment
def _residualpath(inputs, filter_size, path_number, actfn):
    """
    Residual block which performs convolution on the encoder side before
    concatenating with the decoder side.
    """
    def block(x, fl):
        cnn1 = Conv2D(filter_size, (3,3), padding='same', activation=actfn, kernel_initializer = 'he_normal')(inputs)
        cnn2 = Conv2D(filter_size, (1,1), padding='same', activation=actfn, kernel_initializer = 'he_normal')(inputs)

        add = Add()([cnn1, cnn2])

        return add

    cnn = block(inputs, filter_size)
    if path_number <= 3:
        cnn = block(cnn, filter_size)
        if path_number <= 2:
            cnn = block(cnn, filter_size)
            if path_number <= 1:
                cnn = block(cnn, filter_size)

    return cnn

# Main multi-resolution UNet network
def multiresunet(optfn, lossfn, actfn, inputsize=(None, None, 1)):
    inputs = Input(inputsize)

    multires1 = _multiresblock(inputs, 32, actfn)
    pool1 = MaxPool2D()(multires1)

    multires2 = _multiresblock(pool1, 64, actfn)
    pool2 = MaxPool2D()(multires2)

    multires3 = _multiresblock(pool2, 128, actfn)
    pool3 = MaxPool2D()(multires3)

    multires4 = _multiresblock(pool3, 256, actfn)
    # drop4 = Dropout(0.5)(multires4) # Added dropout to last two layers
    pool4 = MaxPool2D()(multires4)

    multires5 = _multiresblock(pool4, 512, actfn)
    # drop5 = Dropout(0.5)(multires5) # Added dropout to last two layers
    upsample = UpSampling2D()(multires5)

    residual4 = _residualpath(multires4, 256, 4, actfn) # drop4 maybe?
    concat = Concatenate()([upsample,residual4])

    multires6 = _multiresblock(concat, 256, actfn)
    upsample = UpSampling2D()(multires6)

    residual3 = _residualpath(multires3, 128, 3, actfn)
    concat = Concatenate()([upsample,residual3])

    multires7 = _multiresblock(concat, 128, actfn)
    upsample = UpSampling2D()(multires7)

    residual2 = _residualpath(multires2, 64, 2, actfn)
    concat = Concatenate()([upsample,residual2])

    multires8 = _multiresblock(concat, 64, act)
    upsample = UpSampling2D()(multires8)

    residual1 = _residualpath(multires1, 32, 1, actfn)
    concat = Concatenate()([upsample,residual1])

    multires9 = _multiresblock(concat, 32, actfn)

    sigmoid = Conv2D(1, (1,1), padding='same', activation='sigmoid')(multires9)

    model = Model(inputs, sigmoid)
    # model.compile(optimizer=optfn, loss=[lossfn], metrics=[losses.dice_loss, lossfn])

    return model
