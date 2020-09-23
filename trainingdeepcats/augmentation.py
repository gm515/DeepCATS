"""
Data Augmentation
Author: Gerald M

Augments the training data using Augmentor package. All augmented data is saved
to the temporary data directory created when training the model. Augmentation is
performed on both the image and mask data in tandem, to ensure the same
operations are applied to correpsonding image-mask pairs. A custom operation is
added to perform poisson noise generation on the images using the skimage
package.
"""

import os, glob, random
import Augmentor
from natsort import natsorted
from Augmentor.Operations import Operation
from skimage.util import noise
from PIL import Image
import numpy as np

class RandomNoise(Operation):
    """
    Custom class to add either poisson or gaussian noise to input images using
    Augmentor pipeline.

    Params
    ------
    probability : float
        Probability value in range 0 to 1 to execute the operation.
    """
    def __init__(self, probability):
        Operation.__init__(self, probability)

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        def do(image):
            image = np.array(image)
            if np.max(image) == 0: # Check mask image is empty
                return Image.fromarray(image)
            elif np.array_equal(image/np.max(image), (image/np.max(image)).astype(bool)): # Check if image is actually mask
                return Image.fromarray(image)
            else:
                vals = len(np.unique(image))
                vals = 2 ** np.ceil(np.log2(vals))

                # Generating noise for each unique value in image.
                image = np.random.poisson(image * vals) / float(vals)

                image /= np.max(image)
                image = np.uint8(image*255.)

                return Image.fromarray(image)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images

def augment(dir, n):
    """
    Augmentation pipeline to do the transformations as listed below.

    Params
    ------
    dir : str
        The base directory containing the images and masks data.
    n : int
        The number of transformed copies to make. Each image and mask pair is
        transformed this number of times.
    """
    training_datagen = Augmentor.Pipeline(source_directory=os.path.join(dir,'images'), output_directory='.', save_format='tif')

    poisson_noise = RandomNoise(probability=0.5)

    training_datagen.ground_truth(os.path.join(dir,'masks'))

    # training_datagen.add_operation(poisson_noise)
    training_datagen.rotate_without_crop(probability=0.5, max_left_rotation=360, max_right_rotation=360, expand=False)
    training_datagen.zoom(probability=0.5, min_factor=0.9, max_factor=1.1)
    # training_datagen.flip_left_right(probability=0.5)
    # training_datagen.flip_top_bottom(probability=0.5)
    training_datagen.skew(probability=0.5, magnitude=0.3)
    training_datagen.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=10)
    training_datagen.shear(probability=0.5,  max_shear_left=2, max_shear_right=2)
    # training_datagen.random_contrast(probability=0.5, min_factor=0.3, max_factor=0.8)
    training_datagen.add_operation(poisson_noise)

    training_datagen.sample(n)

    print ('')

def basicaugment(dir):
    """
    Function to do basic rotation and flipping to augment the images 8-fold.
    The transformations that result in duplicate images are ignored.

    Params
    ------
    dir : str
        The base directory containing the images and masks data.
    """
    images_dir = os.path.join(dir,'images/*.tif')
    masks_dir = os.path.join(dir,'masks/*.tif')

    for imagepath, maskpath in zip(natsorted(glob.glob(images_dir)), natsorted(glob.glob(masks_dir))):
        image = Image.open(imagepath)

        image.transpose(Image.ROTATE_90).save(imagepath.replace('.tif', '_rot90.tif'))
        image.transpose(Image.ROTATE_180).save(imagepath.replace('.tif', '_rot180.tif'))
        image.transpose(Image.ROTATE_270).save(imagepath.replace('.tif', '_rot270.tif'))

        image.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM).save(imagepath.replace('.tif', '_rot90_FTB.tif'))
        image.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM).save(imagepath.replace('.tif', '_rot270_FTB.tif'))

        image.transpose(Image.FLIP_LEFT_RIGHT).save(imagepath.replace('.tif', '_rot0_FTB.tif'))
        image.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(imagepath.replace('.tif', '_rot180_FTB.tif'))

        mask = Image.open(maskpath)

        mask.transpose(Image.ROTATE_90).save(maskpath.replace('.tif', '_rot90.tif'))
        mask.transpose(Image.ROTATE_180).save(maskpath.replace('.tif', '_rot180.tif'))
        mask.transpose(Image.ROTATE_270).save(maskpath.replace('.tif', '_rot270.tif'))

        mask.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM).save(maskpath.replace('.tif', '_rot90_FTB.tif'))
        mask.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM).save(maskpath.replace('.tif', '_rot270_FTB.tif'))

        mask.transpose(Image.FLIP_LEFT_RIGHT).save(maskpath.replace('.tif', '_rot0_FTB.tif'))
        mask.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT).save(maskpath.replace('.tif', '_rot180_FTB.tif'))

    print ('')
