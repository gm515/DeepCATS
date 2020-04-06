"""
DeepCATS Script
Deeplearning Counting in Anatomically Targeted Structures
Author: Gerald M

Version (Python 3) using U-Net + Flask web server
This version uses a UNet to perform semantic segmentation of the images.

Oversampling correction chooses to keep the position of the cell in the middle
slice position, rather than the last detected slice position.

Instructions:
1) First start the Flask server for predicting images (flaskunetserver.py)
1) Run from command line with
    ipython deepcats.py -- <*Ch2_Stitched_Sections> -maskpath <*SEGRES.tif> -hempath <*HEMRES.tif> -radius 10 -ncpu 8 -structures <DORsm,DORpm>

"""

################################################################################
## Module import
################################################################################

import argparse
import csv
import cv2
import glob
import json
import fcntl
import math
import os
import psutil
import requests
import sys
import tifffile
import time
import warnings
import numpy as np
import nibabel as nib
import pandas as pd
from datetime import datetime
from skimage.measure import regionprops, label
from PIL import Image
from skimage import io
from natsort import natsorted
from multiprocessing import Pool, Queue

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

################################################################################
## Function definitions
################################################################################

def slack_message(text, channel, username):
    """
    Slack integration to give slack message to chosen channel. Fill in the slack
    hook url below to send to own slack channel.

    Params
    ------
    text : str
        String of text to post.
    channel : str
        String for the channel to post to.
    username : str
        String for the user posting the message.
    """
    # from urllib3 import request
    import json

    post = {"text": "{0}".format(text),
        "channel": "{0}".format(channel),
        "username": "{0}".format(username),
        "icon_url": "https://github.com/gm515/gm515.github.io/blob/master/Images/imperialstplogo.png?raw=true"}

    try:
        json_data = json.dumps(post)
        req = requests.post('https://hooks.slack.com/services/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',
            data=json_data.encode('ascii'),
            headers={'Content-Type': 'application/json'})
    except Exception as em:
        print("EXCEPTION: " + str(em))

def distance(a, b):
    """
    Calculate distance between coordinates a and b.

    Params
    ------
    a : tuple
    b : tuple

    Returns
    -------
    out : float
        Squared distance between coordinates a and b.
    """
    return (a[0] - b[0])**2  + (a[1] - b[1])**2

def oversamplecorr(centroids, radius):
    """
    Correction for oversampling given list of centroids.

    Params
    ------
    centroids : dictionary
        Dictionary of centroids where key is slice position and items are lists
        of coordinate positions of detected cells.
    radius : int
        Radius with which to claim cells are overlapping.

    Returns
    -------
    out : dictionary
        Output of dictionary of oversampled corrected cell positions.
    """
    keepcentroids = {}
    overlapcentroids = {}
    i = 0

    # First check if there are more than two layers
    if len(list(centroids.keys())) > 1:
        # Loop through successive layers and identify overlapping cells
        for layer1, layer2 in zip(list(centroids.keys())[:-1], list(centroids.keys())[1:]):
            # First check if layers are successive otherwise you cannot correct
            if layer2-layer1 == 1:
                # Store cell centroids for each layer
                layer1centroids = centroids[layer1]
                layer2centroids = centroids[layer2]

                # Loop through each cell in layer 1 and check if overlapping
                for cell in layer1centroids:
                    # Get a boolean list with True in position of cell in layer 2 if cell in layer 1 overlaps and is the minumum distance
                    distances = np.array([distance(cell, cell2) for cell2 in layer2centroids])
                    mindistance = distances == np.min(distances)
                    withindistance = np.array([distance(cell, cell2)<=radius**2 for cell2 in layer2centroids])
                    overlapping = mindistance&withindistance

                    # First check if cell is already within the overlap dictionary, overlapcentroids
                    overlapkey = [key for key, value in overlapcentroids.items() if cell in value]

                    # If there is a True in the overlapping list, then there is a minimum distance oversampled cell detected
                    if True in overlapping:
                        # If so, only add the paired cell
                        if overlapkey:
                            overlapcentroids.setdefault(overlapkey[0],[]).append(layer2centroids[np.argmax(overlapping)])

                        # Else, add both the new cell and pair to it's own unique dictionary key
                        else:
                            overlapcentroids.setdefault(i,[]).append(cell)
                            overlapcentroids.setdefault(i,[]).append(layer2centroids[np.argmax(overlapping)])
                            # Update counter to keep track of number of overlapped cells in total
                            # Uses this as key
                            i += 1

                    # Only if all overlapping is False and the cell is not detected in overlapcentroids already, then add cell to keep
                    if (not True in overlapping) and (not overlapkey):
                        # If no overlap is detected, then stick cell into keep dictionary
                        keepcentroids.setdefault(cell[2], []).append(cell)
            else:
                layer1centroids = centroids[layer1]
                for cell in layer1centroids:
                    keepcentroids.setdefault(cell[2], []).append(cell)

        # Account for the last layer
        layer2centroids = centroids[layer2]
        for cell in layer2centroids:
            overlapkey = [key for key, value in overlapcentroids.items() if cell in value]
            if overlapkey:
                break
            else:
                keepcentroids.setdefault(cell[2], []).append(cell)

        # Go through each overlapping cell and take the middle cell
        # Stick middle cell into the keep dictionary at the relevant slice position
        for key, overlapcells in overlapcentroids.items():
            midcell = overlapcells[int(len(overlapcells)/2)]
            keepcentroids.setdefault(midcell[2], []).append(midcell)

    else:
        keepcentroids = centroids

    return keepcentroids

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

        sys.stdout.write("\nSlice {0} [{1}] {2}% {3}".format(sliceno, arrow + spaces, int(round(percent * 100)), statustext))
        sys.stdout.flush()

def cellcount(imagequeue, radius, size, circ_thresh, use_medfilt):
    """
    Function which performs cell counting by pulling queued items, unpackaging
    the item into separate constructs, including the image to be counted. This
    function does not return anything, but creates a temporary CSV file which
    is written to as objects are detected.

    Params
    ------
    imagequeue : queue
        Queue which contains the objects to pull and process.
    radius : int
        Radius with which to claim cells are overlapping.
    size : int
        Minumum size of cell to expect in pixels. Set this to a value to
        exclude any objects which are smaller than this size and <10 times this
        size.
    circ_thresh : boolean
        Typically False. This uses the older circularity threshold to refine the
        shape of objects which are detected.
    use_medfilt : boolean
        Typically False. This uses a custom median filter to smooth the image.

    Returns
    -------
    None
    """
    while True:
        item = imagequeue.get()
        if item is None:
            break
        else:
            slice_number, image, hemseg_image, row_idx, col_idx, count_path, name, outdir = item
            centroids = []

            if image.shape[0]*image.shape[1] > (radius*2)**2 and np.max(image) != 0.:

                images_array = []

                image = image.astype(np.float32)
                image = (image-np.min(image))/(np.max(image)-np.min(image))

                # Image.fromarray(np.uint8(image*255)).save('/Users/gm515/Desktop/img/'+str(slice_number)+'.tif')

                orig_shape = image.shape

                newshape = tuple((int( 16 * math.ceil( i / 16. )) for i in orig_shape))
                image = np.pad(image, ((0,np.subtract(newshape,orig_shape)[0]),(0,np.subtract(newshape,orig_shape)[1])), 'constant')

                images_array.append(image)
                images_array = np.array(images_array)
                images_array = images_array[..., np.newaxis]

                image_payload = bytearray(images_array)

                shape_payload = bytearray(np.array(images_array.shape))

                # Make payload for request
                payload = {"image": image_payload, "shape": shape_payload}

                result = None
                while result is None:
                    try:
                        result = requests.post("http://localhost:5000/predict", files=payload).json()
                    except:
                        pass

                # ensure the request was sucessful
                if result["success"]:
                    print(str(slice_number)+" request SUCCESSFUL")
                    # loop over the predictions and display them
                    image = np.array(result["predictions"])
                else:
                    print(str(slice_number)+" request UNSUCCESSFULL")

                image = image[0:orig_shape[0],0:orig_shape[1]]

                # Image.fromarray(np.uint8((image>0.25)*255)).save('/Users/gm515/Desktop/pred/'+str(slice_number)+'.tif')

                # Remove objects smaller than chosen size
                image = label(image>0.5, connectivity=image.ndim) # 0.25

                # Get centroids list as (row, col) or (y, x)
                centroids = [region.centroid for region in regionprops(image) if ((region.area>size) and (region.area<10*size) and (((4 * math.pi * region.area) / (region.perimeter * region.perimeter))>0.7))]

                # Add 1 to slice number to convert slice in index to slice file number
                if row_idx is not None:
                    # Convert coordinate of centroid to coordinate of whole image if mask was used
                    if hemseg_image is not None:
                        coordfunc = lambda celly, cellx : (col_idx[cellx], row_idx[celly], slice_number, int(hemseg_image[celly,cellx]))
                    else:
                        coordfunc = lambda celly, cellx : (col_idx[cellx], row_idx[celly], slice_number)
                else:
                    coordfunc = lambda celly, cellx : (cellx, celly, slice_number)

                # Centroids are currently (row, col) or (y, x)
                # Flip order so (x, y) using coordfunc
                centroids = [coordfunc(int(c[0]), int(c[1])) for c in centroids]

            # Write out results to file
            csv_file = os.path.join(count_path, outdir, str(name)+'_unet_count_INQUEUE.csv')

            # Write out detected centroids to CSV
            # File is locked until writing is complete to prevent writing centroids out of order
            while True:
                try:
                    with open(csv_file, 'a+') as f:
                        # Lock file during writing
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        csv.writer(f, delimiter=',').writerows(centroids)

                        # Unlock file and clsoe out
                        fcntl.flock(f, fcntl.LOCK_UN)
                    break
                except IOError as e:
                    # raise on unrelated IOErrors
                    if e.errno != errno.EAGAIN:
                        raise
                    else:
                        time.sleep(0.1)

            print('Finished - Queue position: '+str(slice_number)+' Structure: '+str(name))

if __name__ == '__main__':
    ################################################################################
    ## User defined parameters via command line arguments
    ################################################################################

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

    if mask_path:
        mask = True
    else:
        mask = False
    if hem_path:
        hem = True
    else:
        hem = False

    print ('User defined parameters')
    print( "Image path: {} \nAnnotation path: {} \nHemisphere path: {} \nStructure list: {} \nOversample: {} \nStart: {} \nEnd: {} \nCustom median donut filter: {} \nCircularity threshold: {} \nXYvox: {} \nZvox: {} \nncpu: {} \nSize: {} \nRadius: {}".format(
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
            ))
    print ('')

    ################################################################################
    ## Initialisation
    ################################################################################

    # Create directory to hold the counts in parent folder of images
    count_path = '/'+os.path.join(*image_path.split(os.sep)[:-1])

    starttime = datetime.now()
    starttime.strftime("%m/%d/%Y, %H:%M:%S")

    outdir = 'deepcats_counts_'+starttime.strftime("%m.%d.%Y")

    if not os.path.exists(os.path.join(count_path, outdir)):
        os.makedirs(os.path.join(count_path, outdir))

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(image_path) if each.endswith('.tif')]
    count_files = natsorted(count_files)
    if number_files[0] != None:
        count_files = count_files[number_files[0]-1:number_files[1]]
    print ('Counting in files: '+count_files[0]+' to '+count_files[-1])

    ################################################################################
    ## Retrieving structures IDs
    ################################################################################

    if mask:
        file, extension = os.path.splitext(mask_path)
        if extension == '.nii':
            seg = nib.load(mask_path).get_data()
        else:
            seg = io.imread(mask_path)
        print ('Loaded segmentation atlas')

    if hem:
        file, extension = os.path.splitext(hem_path)
        if extension == '.nii':
            hemseg = nib.load(hem_path).get_data()
        else:
            hemseg = io.imread(hem_path)
        print ('Loaded hemisphere atlas')

    ids = []
    acr = []
    index = np.array([[],[],[]])
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
                    print (name+' not found -> Removed')
    else:
        ids.extend(['None'])
        acr.extend(['None'])

    ################################################################################
    ## Counting
    ################################################################################
    print ('')

    tstart = time.time()

    structure_index = 0

    ################################################################################
    ## Loop through each slice and count in chosen structure
    ################################################################################
    proceed = True

    if mask:
        index = np.array([[],[],[]])
        index = np.concatenate((index, np.where(np.isin(seg,ids))), axis=1)

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
        print ('Setting up Queue')
        imagequeue = Queue()

        # Start processing images
        print ('Creating threads to process Queue items')
        imageprocess = Pool(ncpu, cellcount, (imagequeue, radius, size, circ_thresh, use_medfilt))
        print ('')

        for slice_number in range(zmin,zmax):
            # Load image and convert to dtype=float and scale to full 255 range
            # image = Image.open(image_path+'/'+count_files[slice_number], 'r')
            # temp_size = image.size
            # image = np.frombuffer(image.tobytes(), dtype=np.uint8, count=-1).reshape(image.size[::-1])
            image = tifffile.imread(image_path+'/'+count_files[slice_number], key=0).astype(np.float32)
            temp_size = image.shape[::-1]
            image_max = np.max(image)

            if mask:
                # Get annotation image for slice
                mask_image = np.array(Image.fromarray(seg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))

            # Initiate empty lists
            row_idx = []
            col_idx = []

            ################################################################################
            ## Loop through slices based on cropped boundaries and store into one array
            ################################################################################
            row_idx_array = None
            col_idx_array = None
            # pxvolume = 0

            # Loop through structures available in each slice
            for name, structure in zip(acr,ids):
                # If masking is not required, submit to queue with redundat variables
                if not mask:
                    imagequeue.put((slice_number, image, [None], [None], [None], count_path, name, outdir))
                    print (image_path.split(os.sep)[-3]+' Added slice: '+str(slice_number)+' Queue position: '+str(slice_number-zmin)+' Structure: '+str(name)+' [Memory info] Usage: '+str(psutil.virtual_memory().percent)+'% - '+str(int(psutil.virtual_memory().used*1e-6))+' MB\n')
                else:
                    start = time.time()
                    if structure in mask_image:
                        # Check memory use doesn't go above 80%, otherwise wait
                        memorycheck = False
                        while not memorycheck:
                            if psutil.virtual_memory().percent < 60.0:
                                memorycheck = True
                            else:
                                print ('Warning! Memory too high. Waiting for memory release.')
                                time.sleep(3)

                        # Resize mask
                        mask_image_per_structure = np.copy(mask_image)
                        mask_image_per_structure[mask_image_per_structure!=structure] = 0

                        # Use mask to get global coordinates
                        idx = np.ix_(mask_image_per_structure.any(1),mask_image_per_structure.any(0))
                        row_idx = idx[0].flatten()
                        col_idx = idx[1].flatten()

                        # Apply crop to image and mask, then apply mask
                        image_per_structure = np.copy(image)[idx]
                        mask_image_per_structure = mask_image_per_structure[idx]

                        #image_per_structure = image_per_structure.astype(float)
                        image_per_structure *= 255./image_max

                        mask_image_per_structure = cv2.medianBlur(np.array(mask_image_per_structure).astype(np.uint8), 121) # Apply median filter to massively reduce box like boundary to upsized mask

                        image_per_structure[mask_image_per_structure==0] = 0
                        # image_per_structure = image_per_structure[mask_image_per_structure>0]

                        # Keep track of pixel volume
                        # pxvolume += mask_image_per_structure.any(axis=-1).sum()

                        mask_image_per_structure = None

                        if hem:
                            hemseg_image_per_structure = np.array(Image.fromarray(hemseg[slice_number]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))
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

        print ('')
        print ('Finished queue processing')
        print ('')
        print ('Performing oversampling correction...')

        for name in acr:
            print (name+' oversampling correction')
            with open(os.path.join(count_path, outdir, str(name)+'_unet_count_INQUEUE.csv')) as csvDataFile:
                csvReader = csv.reader(csvDataFile)
                centroids = {}
                for row in csvReader:
                    centroids.setdefault(int(row[2]), []).append([int(entry) for entry in row])

            print (str(sum(map(len, centroids.values())))+' Original uncorrected count')
            keepcentroids = oversamplecorr(centroids,radius)

            with open(os.path.join(count_path, outdir, str(name)+'_unet_count.csv'), 'w+') as f:
                for key in sorted(keepcentroids.keys()):
                    if len(keepcentroids[key]) > 0:
                        csv.writer(f, delimiter=',').writerows([val for val in keepcentroids[key]])

            os.remove(os.path.join(count_path, outdir, str(name)+'_unet_count_INQUEUE.csv'))

        # Oversampling correction and table write-out
        df = pd.DataFrame(columns = ['ROI', 'L', 'R'])

        for file in glob.glob(os.path.join(count_path, outdir, '*_unet_count.csv')):
            keepcentroids = pd.read_csv(file, names=['X', 'Y', 'Z', 'Hemisphere'])
            leftcells = len(keepcentroids.loc[keepcentroids['Hemisphere']==0])
            rightcells = len(keepcentroids.loc[keepcentroids['Hemisphere']==1])

            name = os.path.basename(file).replace('_unet_count.csv', '')

            print (name+', '+str(leftcells+rightcells)+' Final corrected count, L: '+str(leftcells)+' R: '+str(rightcells))

            df = df.append({'ROI':name, 'L':leftcells, 'R':rightcells}, ignore_index=True)

        # Write dataframe to csv
        with open(os.path.join(count_path, outdir, '_counts_table.csv'), 'w') as f:
            df.to_csv(f, index=False)

        print ('')

    print ('~Fin~')
    print (count_path)

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = 'Counting in %s completed in %02d:%02d:%02d:%02d' %(image_path.split(os.sep)[-3], days, hours, minutes, seconds)

    print (text)
    slack_message(text, '#cctn', 'CCTN')
