import argparse
import csv
import cv2
import glob
import json
import os
import psutil
import sys
import tifffile
import time
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime
from PIL import Image
from skimage import io
from natsort import natsorted
from multiprocessing import Pool, Queue
from sendslack import slack_message
from celldetection import cellcount
from oversample import oversamplecorr

"""
DeepCATS Script
Deeplearning Counting in Anatomically Targeted Structures
Author: Gerald M

Version (Python 3) using U-Net + Flask web server
This version uses a UNet to perform semantic segmentation of the images.

Oversampling correction chooses to keep the position of the cell in the middle
slice position, rather than the last detected slice position.

Reads in -1, 0 and +1 image positions in slice for pseudo average effect.

Instructions:
1) First start the Flask server for predicting images (flaskunetserver.py)
1) Run from command line with
    ipython deepcats.py -- <*Ch2_Stitched_Sections> -maskpath <*SEGRES.tif> -hempath <*HEMRES.tif> -radius 10 -ncpu 8 -structures <DORsm,DORpm>

"""


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

"""
Function definitions
"""


def get_children(json_obj, acr, ids):
    for obj in json_obj:
        if obj['children'] == []:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
        else:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
            get_children(obj['children'], acr, ids)
    return (acr, ids)


def get_structure(json_obj, acronym):
    found = (False, None)
    for obj in json_obj:
        if obj['acronym'].lower() == acronym:
            [acr, ids] = get_children(obj['children'], [], [])
            if ids == []:
                acr = [obj['acronym']]
                ids = [obj['id']]
                return (True, acr, ids)
            else:
                acr.append(obj['acronym'])
                ids.append(obj['id'])
                return (True, acr, ids)
        else:
            found = get_structure(obj['children'], acronym)
            if found:
                return found


def progressBar(sliceno, value, endvalue, statustext, bar_length=50):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)) + '/'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\nSlice {0} [{1}] {2}% {3}".format(sliceno, arrow+spaces,
                     int(round(percent * 100)), statustext))
    sys.stdout.flush()


"""
Main function
"""


if __name__ == '__main__':
    """
    User defined parameters via command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('imagepath', default=[], type=str, help='Image directory path for counting')
    parser.add_argument('-maskpath', default=[], type=str, dest='maskpath', help='Annotation file path for masking')
    parser.add_argument('-hempath', default=[], type=str, dest='hempath', help='Hemisphere file path for hemisphere classification')
    parser.add_argument('-structures', default=[], type=str, dest='structures', help='List of structures to count within')
    parser.add_argument('-oversample', action='store_false', default=True, dest='oversample', help='Oversample correction')
    parser.add_argument('-start', default=None, type=int, dest='start', help='Start image number if required')
    parser.add_argument('-end', default=None, type=int, dest='end', help='End image number if required')
    parser.add_argument('-medfilt', default=False, action='store_true', dest='medfilt', help='Use custom median donut filter')
    parser.add_argument('-circthresh', default=0.7, type=float, dest='circthresh', help='Circularity threshold value')
    parser.add_argument('-xyvox', default=0.54, type=float, dest='xyvox', help='XY voxel size')
    parser.add_argument('-zvox', default=10., type=float, dest='zvox', help='Z voxel size')
    parser.add_argument('-ncpu', default=6, type=int, dest='ncpu', help='Number of CPUs to use')
    parser.add_argument('-size', default=100., type=float, dest='size', help='Approximate radius of detected objects')
    parser.add_argument('-radius', default=6, type=float, dest='radius', help='Approximate radius of detected objects')
    parser.add_argument('-downsize', default=1., type=float, dest='downsize', help='Specify if downsizing input images (1um=0.54)')

    args = parser.parse_args()

    image_path = args.imagepath
    mask_path = args.maskpath
    hem_path = args.hempath
    structure_list = args.structures
    over_sample = args.oversample
    number_files = [None, None]
    number_files[0] = args.start
    number_files[1] = args.end
    use_medfilt = args.medfilt
    circ_thresh = args.circthresh
    xyvox = args.xyvox
    zvox = args.zvox
    ncpu = args.ncpu
    size = args.size
    radius = args.radius
    downsize = args.downsize

    if mask_path:
        mask = True
    else:
        mask = False
    if hem_path:
        hem = True
    else:
        hem = False

    print("User defined parameters")
    execparams = "Image path: {} \nAnnotation path: {} \nHemisphere path: {} \nStructure list: {} \nOversample: {} \nStart: {} \nEnd: {} \nCustom median donut filter: {} \nCircularity threshold: {} \nXYvox: {} \nZvox: {} \nncpu: {} \nSize: {} \nRadius: {}".format(
            image_path,
            mask_path,
            hem_path,
            structure_list,
            over_sample,
            number_files[0],
            number_files[1],
            use_medfilt,
            circ_thresh,
            xyvox,
            zvox,
            ncpu,
            size,
            radius
            )
    print(execparams)
    print('')

    """
    Initialisation
    """

    # Create directory to hold the counts in parent folder of images
    count_path = '/'+os.path.join(*image_path.split(os.sep)[:-1])

    starttime = datetime.now()
    starttime.strftime("%m/%d/%Y, %H:%M:%S")

    outdir = 'deepcats_counts_'+starttime.strftime("%m.%d.%Y")

    # Create folder for output files
    if not os.path.exists(os.path.join(count_path, outdir)):
        os.makedirs(os.path.join(count_path, outdir))

    # Write out the parameters to file
    with open(os.path.join(count_path, outdir, 'paramfile.txt'), 'w') as paramfile:
        print(execparams, file=paramfile)

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(image_path) if each.endswith('.tif')]
    count_files = natsorted(count_files)
    if number_files[0] is not None:
        count_files = count_files[number_files[0]-1:number_files[1]]
    print('Counting in files: '+count_files[0]+' to '+count_files[-1])

    """
    Retrieving structures IDs
    """

    if mask:
        file, extension = os.path.splitext(mask_path)
        if extension == '.nii':
            seg = nib.load(mask_path).get_data()
        else:
            seg = io.imread(mask_path)
        print('Loaded segmentation atlas')

    if hem:
        file, extension = os.path.splitext(hem_path)
        if extension == '.nii':
            hemseg = nib.load(hem_path).get_data()
        else:
            hemseg = io.imread(hem_path)
        print('Loaded hemisphere atlas')

    ids = []
    acr = []
    index = np.array([[], [], []])
    if mask:
        anno_file = json.load(open('2017_annotation_structure_info.json'))
        structure_list = [x.strip() for x in structure_list.lower().split(",")]
        for elem in structure_list:
            a, i = get_structure(anno_file['children'], elem)[1:]
            for name, structure in zip(a, i):
                if structure in seg:
                    index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)
                    ids.append(structure)
                    acr.append(name)
                else:
                    print(name+' not found -> Removed')
    else:
        ids.extend(['None'])
        acr.extend(['None'])

    """
    Start counting
    """
    print('')

    tstart = time.time()

    structure_index = 0

    """
    Loop through each slice and count in chosen structure
    """
    proceed = True

    if mask:
        index = np.array([[], [], []])
        index = np.concatenate((index, np.where(np.isin(seg, ids))), axis=1)

        if index.size > 0:
            zmin = int(index[0].min())
            zmax = int(index[0].max())
        else:
            proceed = False
    else:
        zmin = 0
        zmax = len(count_files)

    if proceed:
        # Create a Queue and push images to queue
        print('Setting up Queue')
        imagequeue = Queue()

        # Start processing images
        print('Creating threads to process Queue items')
        imageprocess = Pool(ncpu, cellcount, (imagequeue, radius, size, circ_thresh, use_medfilt))
        print('')

        for slice_number in range(zmin, zmax):
            # Load image and convert to dtype=float and scale to full 255 range
            # image = Image.open(image_path+'/'+count_files[slice_number], 'r')
            # process_size = image.size
            # image = np.frombuffer(image.tobytes(), dtype=np.uint8, count=-1).reshape(image.size[::-1])

            # Load image -1, 0, +1 and max project
            image = tifffile.imread(image_path+'/'+count_files[slice_number], key=0).astype(np.float32)
            # image = np.maximum(image, tifffile.imread(image_path+'/'+count_files[slice_number-1], key=0).astype(np.float32))
            # image = np.maximum(image, tifffile.imread(image_path+'/'+count_files[slice_number+1], key=0).astype(np.float32))

            # PIL.Image.size -> (cols, rows)
            # tifffile.shape -> (rows, cols)

            orig_size = image.shape[::-1]
            process_size = tuple(int(x*downsize) for x in image.shape[::-1])
            image_max = np.max(image)

            image = np.array(Image.fromarray(image).resize(process_size, Image.LANCZOS))

            if mask:
                # Get annotation image for slice
                mask_image = np.array(Image.fromarray(seg[slice_number]).resize(process_size, Image.NEAREST))

            # Initiate empty lists
            row_idx = []
            col_idx = []

            """
            Loop through slices based on cropped boundaries and store into array
            """
            row_idx_array = None
            col_idx_array = None
            # pxvolume = 0

            # Loop through structures available in each slice
            for name, structure in zip(acr, ids):
                # If masking is not required
                # Submit to queue with redundant variables
                if not mask:
                    imagequeue.put((slice_number, image, [None], [None], [None], count_path, name, outdir))
                    print(image_path.split(os.sep)[-3]+' Added slice: '+str(slice_number)+' Queue position: '+str(slice_number-zmin)+' Structure: '+str(name)+' [Memory info] Usage: '+str(psutil.virtual_memory().percent)+'% - '+str(int(psutil.virtual_memory().used*1e-6))+' MB\n')
                else:
                    start = time.time()
                    if structure in mask_image:
                        # Check memory use doesn't go above 80%, otherwise wait
                        memorycheck = False
                        while not memorycheck:
                            if psutil.virtual_memory().percent < 60.0:
                                memorycheck = True
                            else:
                                print('Warning! Memory too high. Waiting for memory release.')
                                time.sleep(3)

                        # Resize mask
                        mask_image_per_structure = np.copy(mask_image)
                        mask_image_per_structure[mask_image_per_structure != structure] = 0

                        # Use mask to get global coordinates
                        idx = np.ix_(mask_image_per_structure.any(1), mask_image_per_structure.any(0))
                        row_idx = idx[0].flatten()
                        col_idx = idx[1].flatten()

                        # Apply crop to image and mask, then apply mask
                        image_per_structure = np.copy(image)[idx]
                        mask_image_per_structure = mask_image_per_structure[idx]

                        # image_per_structure = image_per_structure.astype(float)
                        image_per_structure *= 255./image_max

                        # Apply median filter to massively reduce box like boundary to upsized mask
                        mask_image_per_structure = cv2.medianBlur(np.array(mask_image_per_structure).astype(np.uint8), 121)

                        image_per_structure[mask_image_per_structure == 0] = 0
                        # image_per_structure = image_per_structure[mask_image_per_structure>0]

                        # Keep track of pixel volume
                        # pxvolume += mask_image_per_structure.any(axis=-1).sum()

                        mask_image_per_structure = None

                        if hem:
                            hemseg_image_per_structure = np.array(Image.fromarray(hemseg[slice_number]).resize(process_size, Image.NEAREST))
                            hemseg_image_per_structure = hemseg_image_per_structure[idx]

                        # Add queue number, image, row and col idx to queue
                        imagequeue.put((slice_number, image_per_structure, hemseg_image_per_structure, row_idx, col_idx, count_path, name, outdir))

                        image_per_structure = None
                        hemseg_image_per_structure = None

                        statustext = image_path.split(os.sep)[-3]+' Added slice: '+str(slice_number)+' Queue position: '+str(slice_number-zmin)+' Structure: '+str(name)+' [Memory info] Usage: '+str(psutil.virtual_memory().percent)+'% - '+str(int(psutil.virtual_memory().used*1e-6))+' MB\n'

                        progressBar(slice_number, slice_number-zmin, zmax-zmin, statustext)

        for close in range(ncpu):
            imagequeue.put(None)

        imageprocess.close()
        imageprocess.join()

        print('')
        print('Finished queue processing')
        print('')
        print('Performing oversampling correction...')

        for name in acr:
            print(name+' oversampling correction')
            with open(os.path.join(count_path, outdir, str(name)+'_unet_count_INQUEUE.csv')) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                centroids = {}
                for row in csvReader:
                    centroids.setdefault(int(row[2]), []).append([int(entry) for entry in row])

            print(str(sum(map(len, centroids.values())))+' Original uncorrected count')
            keepcentroids = oversamplecorr(centroids, radius)

            with open(os.path.join(count_path, outdir, str(name)+'_unet_count.csv'), 'w+') as f:
                for key in sorted(keepcentroids.keys()):
                    if len(keepcentroids[key]) > 0:
                        csv.writer(f, delimiter=',').writerows([val for val in keepcentroids[key]])

            # os.remove(os.path.join(count_path, outdir, str(name)+'_unet_count_INQUEUE.csv'))

        # Oversampling correction and table write-out
        df = pd.DataFrame(columns=['ROI', 'L', 'R'])

        for file in glob.glob(os.path.join(count_path, outdir, '*_unet_count.csv')):
            keepcentroids = pd.read_csv(file, names=['X', 'Y', 'Z', 'Hemisphere'])
            leftcells = len(keepcentroids.loc[keepcentroids['Hemisphere'] == 0])
            rightcells = len(keepcentroids.loc[keepcentroids['Hemisphere'] == 1])

            name = os.path.basename(file).replace('_unet_count.csv', '')

            print(name+', '+str(leftcells+rightcells)+' Final corrected count, L: '+str(leftcells)+' R: '+str(rightcells))

            df = df.append({'ROI': name, 'L': leftcells, 'R': rightcells}, ignore_index=True)

        # Remove any duplicate cells
        df = df.drop_duplicates()

        # Write dataframe to csv
        with open(os.path.join(count_path, outdir, '_counts_table.csv'), 'w') as f:
            df.to_csv(f, index=False)

        print('')

    print('~Fin~')
    print(count_path)

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = 'Counting in %s completed in %02d:%02d:%02d:%02d' % (image_path.split(os.sep)[-3], days, hours, minutes, seconds)

    print(text)
    # slack_message(text, '#cctn', 'CCTN')
