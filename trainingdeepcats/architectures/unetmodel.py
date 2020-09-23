"""
The Nested Unet Unet Model
Author: Gerald M

Adapted from paper [UNet++: A Nested U-Net Architecture for Medical Image
Segmentation](arXiv:1807.10165)

Seems slower and also doesn't train as well as GM version(?).
"""

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, ReLU, UpSampling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
import losses

# Dropout fraction
# 0.5 is too high and model does not converge
# 0.2 works well but does not converge past 0.3 dice score
drop=0.1

def standard_unit(input_tensor, stage, nb_filter, actfn, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=actfn, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same')(input_tensor)
    x = Dropout(drop, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=actfn, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same')(x)
    x = Dropout(drop, name='dp'+stage+'_2')(x)

    return x

def unet(optfn, lossfn, actfn, inputsize=(None, None, 1)):
    img_input = Input(inputsize)

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=32, actfn=actfn)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=64, actfn=actfn)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=128, actfn=actfn)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=256, actfn=actfn)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=512, actfn=actfn)

    up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=256, actfn=actfn)

    up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=3)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=128, actfn=actfn)

    up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=3)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=64, actfn=actfn)

    up1_5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=3)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=32, actfn=actfn)

    unet_output = Conv2D(1, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same')(conv1_5)

    model = Model(inputs=img_input, outputs=unet_output)

    # model.compile(optimizer=optfn, loss=lossfn, metrics=[losses.dice_loss, lossfn])

    return model
