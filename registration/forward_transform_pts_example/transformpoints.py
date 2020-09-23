"""
Transform Points and Target Registration Error (TRE)
Author: Gerald M

Importantly this executes as follows. Elastix finds the coordinate mapping from
the fixed image domain to the moving image domain - meaning the moving image is
deformed, but the transformation is defined from fixed to moving image.

Therefore, when registering with a non-deformed fixed and a deformed moving
image, the transformation applied to the moving image allows it to be warped to
match the fixed image. The same transformation map however, only tell you the
fixed points in the moving points domain. So we can appy the transformation to
the fixed image points and see how the transformation aligns the landmarks.

Run in command line with:
    ipython transformpoints.py
"""

import os
import argparse
import datetime
import glob
import time
import shutil
import numpy as np
import SimpleITK as sitk
from skimage import io

import matplotlib.pyplot as plt

# Load images
# Moving is transformed to match fixed image
fixedimg = io.imread('indy.tif')
movingimg = io.imread('indy_deformed.tif')

# Load points
# These points are correpsonding labelled points between fixed and moving
# not necessary for registration but good for testing accuracy etc
fixedpts = np.genfromtxt('indy_points.pts', skip_header=2)
movingpts = np.genfromtxt('indy_deformed_points.pts', skip_header=2)

# First plot out the image
plt.close()
fig, axs = plt.subplots(2, 2)
fig.suptitle('Transformation of sparse points')
axs[0,0].title.set_text('IF(x)')
axs[0,0].imshow(fixedimg, cmap='gray')
axs[0,0].scatter(fixedpts[:,0], fixedpts[:,1], c='r', marker='+')
axs[0,0].axis('off')

axs[0,1].title.set_text('IM(x)')
axs[0,1].imshow(movingimg, cmap='gray')
axs[0,1].scatter(movingpts[:,0], movingpts[:,1], c='b', marker='+')
axs[0,1].axis('off')

# Apply Elastix transformation and save resulting parameter map results
# The following block applies the actual registration process
# Change boolean to True to run
if True:
    SimpleElastix = sitk.ElastixImageFilter()
    SimpleElastix.LogToFileOn()
    # Set fixed and moving image
    SimpleElastix.SetFixedImage(sitk.GetImageFromArray(fixedimg))
    SimpleElastix.SetMovingImage(sitk.GetImageFromArray(movingimg))

    # Create a vector of the parameter map which stores the registration parameters
    # Here we just need affine and bspline transform to get a good result
    parameterMapVector = sitk.VectorOfParameterMap()
    affineParameterMap = sitk.ReadParameterFile('paraffine.txt')
    parameterMapVector.append(affineParameterMap)
    bsplineParameterMap = sitk.ReadParameterFile('parbspline.txt')
    parameterMapVector.append(bsplineParameterMap)
    SimpleElastix.SetParameterMap(parameterMapVector)

    # Print parameter map which is just a sanity check to make sure parameters
    # are set
    SimpleElastix.PrintParameterMap()

    # Execute
    # This is the actual execution of Elastix and uses the parameter map
    SimpleElastix.Execute()

    # Get transform map
    # After completing the transformation, store the combined transformation
    # which was required. This transformation can be used to transform another
    # image or to transform spare points as below.
    transformMap = SimpleElastix.GetTransformParameterMap()

# Fetch pre-saved transformation files
# This transforms the sparse points
# Change boolean to True to run
if False:
    transformMap = sitk.VectorOfParameterMap()
    affineParameterMap = sitk.ReadParameterFile('TransformParameters.0.txt')
    transformMap.append(affineParameterMap)
    bsplineParameterMap = sitk.ReadParameterFile('TransformParameters.1.txt')
    transformMap.append(bsplineParameterMap)

resData = sitk.Transformix(sitk.GetImageFromArray(movingimg), transformMap)
sitk.WriteImage(resData, 'transformed_moving_result.tif')
resData = sitk.GetArrayFromImage(resData)

# Transform the data points
TransformixTransform = sitk.TransformixImageFilter()
TransformixTransform.SetMovingImage(sitk.GetImageFromArray(movingimg))
TransformixTransform.SetTransformParameterMap(transformMap)
TransformixTransform.SetFixedPointSetFileName('indy_points.pts')
TransformixTransform.Execute()

# Get transformed points
transformedpts = np.genfromtxt('outputpoints.txt')[:,41:43]

# Plot initially transformed image
axs[1,0].title.set_text('IM(T(x))')
axs[1,0].imshow(resData, cmap='gray')
axs[1,0].axis('off')

# Plot transforme fixed points to moving space
axs[1,1].title.set_text('IM(x)+pts')
axs[1,1].imshow(movingimg, cmap='gray')
axs[1,1].scatter(movingpts[:,0], movingpts[:,1], c='b', marker='+')
axs[1,1].scatter(transformedpts[:,0], transformedpts[:,1], c='r', marker='+')
axs[1,1].axis('off')

# Calculate TRE (Target Registration Error)
# Initial
initialerrors = [np.linalg.norm(moving-fixed) for moving, fixed in zip(movingpts, fixedpts)]
initialmean = np.mean(initialerrors)
initialstd = np.std(initialerrors)
# Target transformation
targetregerrors = [np.linalg.norm(moving-fixed) for moving, fixed in zip(movingpts, transformedpts)]
targetregmean = np.mean(targetregerrors)
targetregstd = np.std(targetregerrors)

print ("==================================")
print ("             Results              ")
print ("Initial")
print ("    Mean = {:.2f} ± {:.2f} (std) pixels".format(initialmean, initialstd))
print ("Target registration error")
print ("    Mean = {:.2f} ± {:.2f} (std) pixels".format(targetregmean, targetregstd))
print ("==================================")
