"""
Data Preprocessing
Author: Gerald M

Pre-processes the raw data stored in data_copy/ by splitting the into 70%
training and 30% validation/testing. Data augmentation is performed using
augmentation.py module.

All data following augmentation is normalised with,

    (img-np.min(img))/(np.max(img)-np.min(img))

and NaN values detected. As NaN can propegate during training, exceptions are
called if these values exist in the training and validation data.
"""

import Augmentor
import datetime
import glob
import os
import random
import shutil
import cleanup
import augmentation
import pickle
from distutils.dir_util import copy_tree
from PIL import Image
from natsort import natsorted
import numpy as np

def preprocess(dir, fn):
    """
    Data preprocessing function. Given input directory, image and mask data is
    copied into a new temporary working directory called "data_copy". Data from
    the input directory is copied to "data_copy" and processed.

    Parameters
    ----------
    dir : str
        String for the directory to perform training with.
    n : int
        The factor of augmented images to generate with Augmentor.

    Returns
    -------
    None : All processed images are saved into the data_copy directory.

    """

    # Insert condition to check whether the augmentation generates NaN data
    # If so, repeat the entire process again...
    nandetected = False

    while not nandetected:
        data_copy_dir = dir.strip('/')+'_WORKINGCOPY'

        training_data_dir = os.path.join(data_copy_dir, 'training_data')
        test_data_dir = os.path.join(data_copy_dir, 'test_data')

        # Clean up any old directories and create new directories
        cleanup.clean()

        os.makedirs(os.path.join(training_data_dir, 'images'))
        os.makedirs(os.path.join(training_data_dir, 'masks'))

        os.makedirs(os.path.join(test_data_dir, 'images'))
        os.makedirs(os.path.join(test_data_dir, 'masks'))

        # Make a working directory copy of data so we don't lose anything
        os.makedirs(os.path.join(data_copy_dir, 'images'))
        os.makedirs(os.path.join(data_copy_dir, 'masks'))
        copy_tree(os.path.join(dir, 'images'), os.path.join(data_copy_dir, 'images'))
        copy_tree(os.path.join(dir, 'masks'), os.path.join(data_copy_dir, 'masks'))

        print ('Performing basic rotation and flipping augmentation on data copy...')

        augmentation.basicaugment(data_copy_dir)

        print ('Done!')

        print ('Performing augmentation on data copy...')

        if fn>0:
            images_data = os.listdir(data_copy_dir+'/images/')
            n=fn*len(images_data)
            augmentation.augment(data_copy_dir,n)

            aug_images = glob.glob(data_copy_dir+'/images/images_original*')
            aug_masks = glob.glob(data_copy_dir+'/images/_groundtruth*')
            aug_images.sort(key=lambda x:x[-40:])
            aug_masks.sort(key=lambda x:x[-40:])

            for i, (image_file, mask_file) in enumerate(zip(aug_images, aug_masks)):
                shutil.move(image_file, image_file.replace('images_original_', ''))
                shutil.move(mask_file, mask_file.replace('_groundtruth_(1)_images_', '').replace('/images/', '/masks/'))

            print ('Augmented and saved with n='+str(n)+' samples!')

        print ('Randomly selecting/moving 70% training and 30% test data...')
        images_data = natsorted(os.listdir(data_copy_dir+'/images/'))
        masks_data = natsorted(os.listdir(data_copy_dir+'/masks/'))

        # Changed the sampling so they sample approximately the same distribution
        # Now sampling is 75:25
        test_images_data = images_data[::4]
        test_masks_data = [f.replace('/images/', '/masks/') for f in test_images_data]
        training_images_data = [x for x in images_data if x not in test_images_data]
        training_masks_data = [f.replace('/images/', '/masks/') for f in training_images_data]

        # Old random sampling method for 70:30 data split
        # random.shuffle(images_data)
        # training_images_data = images_data[:int(0.7*len(images_data))]
        # training_masks_data = [f.replace('/images/', '/masks/') for f in training_images_data]
        # test_images_data  = images_data[int(0.7*len(images_data)):]
        # test_masks_data = [f.replace('/images/', '/masks/') for f in test_images_data]

        for f in training_images_data:
            shutil.copy(os.path.join(data_copy_dir,'images',f), os.path.join(training_data_dir,'images',f))

        for f in training_masks_data:
            shutil.copy(os.path.join(data_copy_dir,'masks',f), os.path.join(training_data_dir,'masks',f))

        for f in test_images_data:
            shutil.copy(os.path.join(data_copy_dir,'images',f), os.path.join(test_data_dir,'images',f))

        for f in test_masks_data:
            shutil.copy(os.path.join(data_copy_dir,'masks',f), os.path.join(test_data_dir,'masks',f))

        print ('Done!')

        training_data_images = []
        training_data_masks = []
        test_data_images = []
        test_data_masks = []

        print ('Loading data...')

        for imagepath, maskpath in zip(natsorted(glob.glob(training_data_dir+'/images/*')), natsorted(glob.glob(training_data_dir+'/masks/*'))):
            image = Image.open(imagepath).resize((512, 512), resample=Image.BILINEAR)
            mask = Image.open(maskpath).resize((512, 512), resample=Image.NEAREST)
            training_data_images.append(np.array(image))
            training_data_masks.append(np.array(mask))

        for imagepath, maskpath in zip(natsorted(glob.glob(test_data_dir+'/images/*')), natsorted(glob.glob(test_data_dir+'/masks/*'))):
            image = Image.open(imagepath).resize((512, 512), resample=Image.BILINEAR)
            mask = Image.open(maskpath).resize((512, 512), resample=Image.NEAREST)
            test_data_images.append(np.array(image))
            test_data_masks.append(np.array(mask))

        training_data_images = np.array(training_data_images).astype(np.float32)
        training_data_masks = np.array(training_data_masks).astype(np.float32)
        test_data_images = np.array(test_data_images).astype(np.float32)
        test_data_masks = np.array(test_data_masks).astype(np.float32)

        print ('Done!')

        print ('Running normalisation...')

        for idx, img in enumerate(training_data_images):
            training_data_images[idx] = (img-np.min(img))/(np.max(img)-np.min(img))

        for idx, img in enumerate(training_data_masks):
            if np.sum(img) > 0:
                img[img < (np.min(img)+np.max(img))/2] = 0.
                img[img >= (np.min(img)+np.max(img))/2] = 1.
                training_data_masks[idx] = img

        for idx, img in enumerate(test_data_images):
            test_data_images[idx] = (img-np.min(img))/(np.max(img)-np.min(img))

        for idx, img in enumerate(test_data_masks):
            if np.sum(img) > 0:
                img[img < (np.min(img)+np.max(img))/2] = 0.
                img[img >= (np.min(img)+np.max(img))/2] = 1.
                test_data_masks[idx] = img

        print ('Done!')

        print ('Checking nan...')

        if np.isnan(training_data_images).any() or np.isnan(training_data_masks).any() or np.isnan(test_data_images).any() or np.isnan(test_data_masks).any():
            print ('NaN value detected. Repeating the augmentation process again...')
        else:
            nandetected = True

    print ('Done!')

    training_data_images = training_data_images[..., np.newaxis]
    training_data_masks = training_data_masks[..., np.newaxis]
    test_data_images = test_data_images[..., np.newaxis]
    test_data_masks = test_data_masks[..., np.newaxis]

    return (training_data_images, training_data_masks, test_data_images, test_data_masks)
