"""
Registration
Author: Gerald M

Run in command line with:
    ipython registration.py -- <autoflpath>
"""

import os
import argparse
import datetime
import time
import shutil
import numpy as np
import SimpleITK as sitk
from skimage import io

if __name__ == '__main__':
    """
    Perform registration of Allen Brain Atlas CCF 3.0 to STP data.

    Parameters
    ----------
    autoflpath : str
        Path to the autofluorescence atlas of STP data - downsized to 10 um
        isotropic voxel size.
    avgpath : str
        Path to the average atlas. Default is the atlas in repo.
    annopath : str
        Path to the annotation atlas. Default is atlas in repo.
    first : optional, int
        If needing to limit the first slice in the average atlas to aid
        registration result.
    last : optional, int
        If needing to limit the last slice in the atlas to aid registration
        result.
    fixedpts : optional, str
        If using fixed points for landmarks, this is the file path. Default
        False.
    movingpts : optional, str
        If using fixed points for landmarks, this is the file path. Default
        False.
    """

    # Command line parameter passer
    parser = argparse.ArgumentParser()

    parser.add_argument('autoflpath', default=[], type=str, help='File path for autofluorescence atlas')
    parser.add_argument('-avgpath', default='atlases/average_10um.tif', dest='avgpath', type=str, help='File path for average atlas')
    parser.add_argument('-annopath', default='atlases/annotation_10um.tif', dest='annopath', type=str, help='File path for annotation atlas')
    parser.add_argument('-first', default=0, type=int, dest='first', help='First slice in average atlas')
    parser.add_argument('-last', default=1320, type=int, dest='last', help='Last slice in average atlas')
    parser.add_argument('-fixedpts', default=False, type=str, dest='fixedpts', help='Path for fixed points')
    parser.add_argument('-movingpts', default=False, type=str, dest='fixedpts', help='path for moving points')

    args = parser.parse_args()

    # Load all the atlases and generate hemisphere atlas
    print ('Loading all atlases...')
    fixedData = sitk.ReadImage(args.autoflpath)
    fixedData.SetSpacing((0.01, 0.01, 0.01))
    print ('Autofluorescence atlas loaded')
    movingData = sitk.GetImageFromArray(io.imread(args.avgpath))
    movingData.SetSpacing((0.01, 0.01, 0.01))
    print ('Average atlas loaded')
    annoData = sitk.GetImageFromArray(io.imread(args.annopath))
    annoData.SetSpacing((0.01, 0.01, 0.01))
    print ('Annotation atlas loaded')
    hemData = np.zeros(movingData.GetSize())
    hemData[570::,:,:] = 1
    hemData = sitk.GetImageFromArray(np.uint8(np.swapaxes(hemData,0,2)))
    hemData.SetSpacing((0.01, 0.01, 0.01))
    print ('Hemisphere atlas generated')

    rigidfile = 'par00GMrigid.txt'
    affinefile = 'par00GMaffine.txt'
    bsplinefile = 'par00GMbspline.txt'

    strdate = datetime.datetime.today().strftime('%Y_%m_%d')

    # Create output directory
    outdir = os.path.join(os.path.dirname(args.autoflpath), 'Registration_'+affinefile+bsplinefile+'_'+strdate)
    try:
        os.mkdir(outdir)
    except OSError:
        print ('Failed to create output directory at {}'.format(outdir))
    else:
        print ('Successfully created output directory at {}'.format(outdir))

    # Copy execution files to output directory
    outexecdir = os.path.join(outdir, 'execfiles')
    try:
        os.mkdir(outexecdir)
        shutil.copyfile('registration.py', outexecdir)
        shutil.copyfile('parameters/', outexecdir)
    except OSError:
        print ('Failed to create copy of execution files at {}'.format(outexecdir))
    else:
        print ('Successfully created copy of execution files at {}'.format(outexecdir))

    tstart = time.time()

    # Initiate SimpleElastix
    SimpleElastix = sitk.ElastixImageFilter()
    SimpleElastix.LogToFileOn()
    SimpleElastix.SetOutputDirectory(outdir)
    SimpleElastix.SetFixedImage(fixedData)
    SimpleElastix.SetMovingImage(movingData)

    # If corresponding points being used, set up inputs
    if args.fixedpts:
        SimpleElastix.SetFixedPointSetFileName(args.fixedpointspath)
        SimpleElastix.SetMovingPointSetFileName(args.movingpointspath)

    # Create segmentation map
    parameterMapVector = sitk.VectorOfParameterMap()

    # Start with Rigid
    rigidParameterMap = sitk.ReadParameterFile(os.path.join('parametermaps', rigidfile))
    parameterMapVector.append(rigidParameterMap)

    # Start with Affine
    affineParameterMap = sitk.ReadParameterFile(os.path.join('parametermaps', affinefile))
    parameterMapVector.append(affineParameterMap)

    # Add BSpline
    bsplineParameterMap = sitk.ReadParameterFile(os.path.join('parametermaps', bsplinefile))
    if args.fixedpts:
        bsplineParameterMap['Metric'] = [bsplineParameterMap['Metric'][0], 'CorrespondingPointsEuclideanDistanceMetric']
    parameterMapVector.append(bsplineParameterMap)

    # Set the parameter map
    SimpleElastix.SetParameterMap(parameterMapVector)

    # Print parameter map
    SimpleElastix.PrintParameterMap()

    # Execute
    SimpleElastix.Execute()

    # Save average transform
    avgSeg = SimpleElastix.GetResultImage()

    # Get transform map and apply to segmentation data and hemisphere
    transformMap = SimpleElastix.GetTransformParameterMap()

    # Loop through the transform maps and set interpolation order to 0 (NNN)
    for i in range(len(transformMap)):
        transformMap[i]['FinalBSplineInterpolationOrder'] = ['0']

    resultSeg = sitk.Transformix(annoData, transformMap)
    hemSeg = sitk.Transformix(hemData, transformMap)

    # Write average transform and segmented results
    sitk.WriteImage(avgSeg, os.path.join(outdir, os.path.basename(args.autoflpath)+'AVGRES.tif'))
    sitk.WriteImage(resultSeg, os.path.join(outdir, os.path.basename(args.autoflpath)+'SEGRES.tif'))
    sitk.WriteImage(hemSeg, os.path.join(outdir, os.path.basename(args.autoflpath)+'HEMRES.tif'))

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = args.autoflpath+'\n SimpleElastix segmentation completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

    print ('')
    print (text)
