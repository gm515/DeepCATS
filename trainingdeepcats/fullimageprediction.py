"""
Full Image Classification
Author: Gerald M

Splits the image into blocks of 512x512 and classifies each block. For interest,
gives time to calculate 1000 images.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import time
import warnings
import numpy as np
from PIL import Image

# Modules for deep learning
from tensorflow.keras.models import model_from_json

from tensorflow.keras import backend as K
# This line must be executed before loading Keras model.
K.set_learning_phase(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

if __name__ == '__main__':
    img_path = '/home/gm515/Documents/RabiesBL-Z55-132.tif'

    print ('Loading image...')
    orig_img = Image.open(img_path)
    orig_img = orig_img.resize(tuple(int(0.5*x) for x in orig_img.size))
    orig_img = np.array(orig_img).astype(np.float32)
    orig_img = (orig_img-np.min(orig_img))/(np.max(orig_img)-np.min(orig_img))
    print ('Done!')

    model_path = '/home/gm515/Documents/machinelearning/models/2020_07_28_Adam0.0001_BCE_elu_GPUs1_Batch8_Aug6_GM_UNetv3/GM_UNetv3_model.json'
    weights_path = '/home/gm515/Documents/machinelearning/models/2020_07_28_Adam0.0001_BCE_elu_GPUs1_Batch8_Aug6_GM_UNetv3/GM_UNetv3_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    print ('Loading model...')
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    print ('Done!')

    tstart = time.time()

    print ('Running 512x512 window through image and predicting...')
    # Split true and image into 512x512 blocks
    imgarray = []
    ws = 512
    for y in range(0,orig_img.shape[0], ws):
        for x in range(0,orig_img.shape[1], ws):
            imagecrop = orig_img[y:y+ws, x:x+ws]
            imagecroppad = np.zeros((ws, ws))
            if (np.max(imagecrop)-np.min(imagecrop))>0: # Ignore any empty data and zero divisions
                imagecroppad[:imagecrop.shape[0],:imagecrop.shape[1]] = (imagecrop-np.min(imagecrop))/(np.max(imagecrop)-np.min(imagecrop))
            imagecroppad = imagecroppad[..., np.newaxis]
            imgarray.append(imagecroppad)

    imgarray = np.array(imgarray)

    print ('Predicting...')
    tstart = time.time()
    predarray = model.predict(imgarray, batch_size=6)
    print (time.time()-tstart)
    pred = np.zeros((int(np.ceil(orig_img.shape[0]/ws)*ws), int(np.ceil(orig_img.shape[1]/ws)*ws)))
    i = 0
    for y in range(0,orig_img.shape[0], ws):
        for x in range(0,orig_img.shape[1], ws):
            pred[y:y+ws, x:x+ws] = np.squeeze(predarray[i])
            i += 1

    telapsed = time.time()-tstart
    hours, rem = divmod(telapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print ("Single image {:0>2} hours {:0>2} minutes {:05.2f} seconds".format(int(hours),int(minutes),seconds))

    hours, rem = divmod(telapsed*1000, 3600)
    minutes, seconds = divmod(rem, 60)
    print ("1000 images  {:0>2} hours {:0>2} minutes {:05.2f} seconds".format(int(hours),int(minutes),seconds))
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(pred_img)
    # plt.show()

    Image.fromarray((pred*255).astype(np.uint8)).save('RabiesBL-Z55-132-pred-20200728.tif')
