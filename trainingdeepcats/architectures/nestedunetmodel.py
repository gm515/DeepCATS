"""
The Nested Unet Model
Author: Gerald M

Adapted from paper [UNet++: A Nested U-Net Architecture for Medical Image
Segmentation](arXiv:1807.10165)

GitHub source https://github.com/CarryHJR/Nested-UNet.git
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU, UpSampling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import losses

def standard_unit(input_tensor, stage, nb_filter, actfn, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=actfn, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same')(input_tensor)
    x = Dropout(0.5, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=actfn, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same')(x)
    x = Dropout(0.5, name='dp'+stage+'_2')(x)

    return x

def nestedunet(optfn, lossfn, actfn, inputsize=(None, None, 1), deep_supervision=False):
    img_input = Input(inputsize)

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=32, actfn=actfn)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=64, actfn=actfn)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    # up1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    up1_2 = UpSampling2D((2,2))(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=32, actfn=actfn)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=128, actfn=actfn)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    # up2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    up2_2 = UpSampling2D((2,2))(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=64, actfn=actfn)

    # up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    up1_3 = UpSampling2D((2,2))(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=32, actfn=actfn)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=256, actfn=actfn)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    # up3_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    up3_2 = UpSampling2D((2,2))(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=128, actfn=actfn)

    # up2_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    up2_3 = UpSampling2D((2,2))(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=64, actfn=actfn)

    # up1_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    up1_4 = UpSampling2D((2,2))(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=32, actfn=actfn)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=512, actfn=actfn)

    # up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    up4_2 = UpSampling2D((2,2))(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=256, actfn=actfn)

    # up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    up3_3 = UpSampling2D((2,2))(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=128, actfn=actfn)

    # up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    up2_4 = UpSampling2D((2,2))(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=64, actfn=actfn)

    # up1_5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    up1_5 = UpSampling2D((2,2))(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=32, actfn=actfn)

    nestnet_output_1 = Conv2D(1, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(1, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same')(conv1_3)
    nestnet_output_3 = Conv2D(1, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same')(conv1_5)

    if deep_supervision:
        model = Model(inputs=img_input, outputs=[nestnet_output_1,
                                                nestnet_output_2,
                                                nestnet_output_3,
                                                nestnet_output_4])
    else:
        model = Model(inputs=img_input, outputs=[nestnet_output_4])

    # model.compile(optimizer=optfn, loss=lossfn, metrics=[losses.dice_loss, lossfn])

    return model
