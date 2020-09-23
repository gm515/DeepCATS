"""
Model Training
Author: Gerald M

Trains the model by using clean-up, pre-processing and augmentation modules.
Can be run from command line with following,

    ipython -- trainmodel.py Adam 1e-4 BCE elu 1 GM_UNet 6

to dictate the following:
- loss function
- learning rate
- optimizer
- activation fuction
- number of GPUs
- model name
- amount of augmentation

Model architecture, weights and training history are saved into a dated
directory under models.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import subprocess
import time
import datetime
import numpy as np
import pandas as pd

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback, Callback
from tensorflow.keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from keras import backend as K

# import custom modules
import cleanup
import preprocessing
import losses

# Import different architectures
from architectures import nestedunetmodel
from architectures import multiresunetmodel
from architectures import unetmodel
from architectures import unetmodelv2
from architectures import unetmodelv3

# Set the training precision to speed up training time..?
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# Alpha scheduler for modifying loss weights
# Added a delay to only start modifying alpha after set number of epochs
class AlphaScheduler(Callback):
    def __init__(self, alpha, delay, function):
        self.alpha = alpha
        self.delay = delay
        self.function = function

    def on_epoch_end(self, epoch, logs=None):
        updated_alpha = self.function(K.get_value(self.alpha), self.delay, epoch)
        K.set_value(self.alpha, updated_alpha)
    #    print("End of epoch {}, alpha={}".format(epoch, self.alpha))


def reduce_alpha(alpha, delay, epoch):
    if epoch < delay:
        val = alpha
    else:
        val = np.clip(alpha - 0.01, 0.01, 1)
    return val


alpha = K.variable(1., dtype='float32')
delay = 0


if __name__ == '__main__':
    if len(sys.argv) > 0:
        opt_arg1 = str(sys.argv[1])
        lr_arg2 = float(sys.argv[2])
        loss_arg3 = str(sys.argv[3])
        act_arg4 = str(sys.argv[4])
        ngpu_arg5 = int(sys.argv[5])
        model_arg6 = str(sys.argv[6])
        aug_arg7 = int(sys.argv[7])

        if opt_arg1 == 'Adam': optimizer = Adam(lr=lr_arg2)
        if opt_arg1 == 'SGD': optimizer = SGD(lr=lr_arg2)

        if loss_arg3 == 'BCE': loss = 'binary_crossentropy' # works best
        if loss_arg3 == 'FTL': loss = losses.focal_tversky # works well but still not as good as BCE
        if loss_arg3 == 'Combo': loss = losses.combo_loss
        if loss_arg3 == 'MSE': loss = 'mean_squared_error'
        if loss_arg3 == 'BL': loss = losses.surface_loss
        if loss_arg3 == 'BCE+BL': loss = losses.bce_surface_loss(alpha)
        if loss_arg3 == 'DSC+BL': loss = losses.dsc_surface_loss(alpha)

        if act_arg4 == 'relu': act = 'relu'
        if act_arg4 == 'elu': act = 'elu'

    modelname = model_arg6

    # Scale batch size to number of GPUs being used
    batch = 8 * ngpu_arg5

    # Choose which data to train on and how many images to augment
    # 4 - works well without any noise augmentation
    # 5 - perhaps will work well with noise augmentation? Does
    # 6 - works best without using all memory (6 transforms so six sets of training data per transform)
    n = aug_arg7
    training_dir = 'data/GM_MG_combined_data_n217'
    train_x, train_y, val_x, val_y = preprocessing.preprocess(training_dir, n)

    # Get today's date for model saving
    strdate = datetime.datetime.today().strftime('%Y_%m_%d')

    savedirpath = os.path.join('models', strdate+'_'+opt_arg1+str(lr_arg2)+'_'+loss_arg3+'_'+act_arg4+'_GPUs'+str(ngpu_arg5)+'_Batch'+str(batch)+'_Aug'+str(aug_arg7)+'_'+model_arg6)
    if not os.path.exists(savedirpath):
        os.makedirs(savedirpath)

    filepath = os.path.join(savedirpath, modelname+'_weights.best.hdf5')

    # Log output to file
    # log_stdout = sys.stdout
    # logfile = open(os.path.join(savedirpath, 'logfile.txt'), 'w')
    # sys.stdout = logfile

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if model_arg6 == 'GM_UNet': model = unetmodelv2.unet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act) # Works best
        if model_arg6 == 'GM_UNetv3': model = unetmodelv3.unet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act) # Works best
        if model_arg6 == 'ZX_UNet': model = unetmodel.unet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act)
        if model_arg6 == 'ZX_NestedUNet': model = nestedunetmodel.nestedunet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act)
        if model_arg6 == 'ZX_MultiResUNet': model = multiresunetmodel.multiresunet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act) # Does not work

        model.compile(optimizer=optimizer, loss=[loss], metrics=[losses.dice_loss, loss])

    # Serialize model to JSON
    modeljson = model.to_json()
    with open(os.path.join(savedirpath, modelname+'_model.json'), 'w') as jsonfile:
        jsonfile.write(modeljson)

    checkpoint = ModelCheckpoint(filepath, monitor='val_dice_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor='val_dice_loss', mode='min', patience=30, verbose=1)
    redonplat = ReduceLROnPlateau(monitor='val_dice_loss', mode='min', patience=10, verbose=1)
    modalpha = AlphaScheduler(alpha, delay, reduce_alpha)
    callbacks_list = [checkpoint, early, redonplat, modalpha]

    # Capture the Git repo status being executed
    gitstr = subprocess.check_output('git log -1'.split()).decode()
    print('Training on following git commit...')
    print(gitstr)

    tstart = time.time()

    history = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=batch,
        epochs=250,
        shuffle=True,
        callbacks=callbacks_list)

    tend = time.time()

    # Write out the training history to file
    pd.DataFrame(history.history).to_csv(os.path.join(savedirpath, 'trainhistory.csv'))

    cleanup.clean()

    # Plot out to see progress
    # import matplotlib.pyplot as plt
    # plt.plot(history.history['dice_loss'])
    # plt.plot(history.history['val_dice_loss'])
    # plt.title('Dice loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper right')
    # plt.show()

    minutes, seconds = divmod(tend-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print('============================================')
    print('')
    print('         Model: {}'.format(model_arg6))
    print('      Data set: {}'.format(training_dir))
    print('Augmentation n: {} w/ training {}, val {}'.format(n, len(train_x), len(val_x)))
    print(' Batch per GPU: {}'.format(batch))
    print('        N GPUs: {}'.format(ngpu_arg5))
    print('     Optimiser: {}'.format(opt_arg1))
    print('          Loss: {}'.format(loss_arg3))
    print('    Activation: {}'.format(act_arg4))
    print(' Learning rate: {}'.format(lr_arg2))
    print('Best val. loss: {:0.4f}'.format(np.min(history.history['val_dice_loss'])))
    print('Execution time: {:0.0f} days {:0.0f} hrs {:0.0f} mins {:0.0f} secs'.format(days, hours, minutes, seconds))
    print('')
    print('============================================')

    # Log output to file
    log_stdout = sys.stdout
    logfile = open(os.path.join(savedirpath, 'logfile.txt'), 'w')
    sys.stdout = logfile

    # Capture the Git repo status being executed
    gitstr = subprocess.check_output('git log -1'.split()).decode()
    print('Training on following git commit...')
    print(gitstr)
    print('')
    print('============================================')
    print('')
    print('         Model: {}'.format(model_arg6))
    print('      Data set: {}'.format(training_dir))
    print('Augmentation n: {} w/ training {}, val {}'.format(n, len(train_x), len(val_x)))
    print(' Batch per GPU: {}'.format(batch))
    print('        N GPUs: {}'.format(ngpu_arg5))
    print('     Optimiser: {}'.format(opt_arg1))
    print('          Loss: {}'.format(loss_arg3))
    print('    Activation: {}'.format(act_arg4))
    print(' Learning rate: {}'.format(lr_arg2))
    print('Best val. loss: {:0.4f}'.format(np.min(history.history['val_dice_loss'])))
    print('Execution time: {:0.0f} days {:0.0f} hrs {:0.0f} mins {:0.0f} secs'.format(days, hours, minutes, seconds))
    print('')
    print('============================================')

    sys.stdout = log_stdout
    logfile.close()
