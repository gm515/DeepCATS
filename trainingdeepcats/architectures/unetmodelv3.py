"""
The Unet Model
Author: Gerald M

Adapted from paper [U-Net: Convolutional Networks for Biomedical Image
Segmentation](arXiv:1505.04597)

Modified version of unetmodelv2
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU, UpSampling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import losses

def unet(optfn, lossfn, actfn, inputsize=(None, None, 1)):
    inputs = Input(inputsize)

    conv1 = Conv2D(32, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    drop1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = Conv2D(64, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    drop2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

    conv3 = Conv2D(128, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

    conv4 = Conv2D(256, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = BatchNormalization()(conv5)

    up6 = Conv2D(256, 2, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = BatchNormalization()(conv6)

    up7 = Conv2D(128, 2, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop7 = BatchNormalization()(conv7)

    up8 = Conv2D(64, 2, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    drop8 = BatchNormalization()(conv8)

    up9 = Conv2D(32, 2, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = actfn, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    drop9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(drop9)

    model = Model(inputs, conv10)

    # model.compile(optimizer=optfn, loss=[lossfn], metrics=[losses.dice_loss, lossfn])

    return model
