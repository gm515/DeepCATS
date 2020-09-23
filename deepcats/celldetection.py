import csv
import fcntl
import math
import os
import requests
import time
import numpy as np
from skimage.measure import regionprops, label


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

                newshape = tuple((int(16 * math.ceil(i/16.)) for i in orig_shape))
                image = np.pad(image, ((0, np.subtract(newshape, orig_shape)[0]), (0, np.subtract(newshape, orig_shape)[1])), 'constant')

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
                    except TypeError:
                        pass

                # ensure the request was sucessful
                if result["success"]:
                    print(str(slice_number)+" request SUCCESSFUL")
                    # loop over the predictions and display them
                    image = np.array(result["predictions"])
                else:
                    print(str(slice_number)+" request UNSUCCESSFULL")

                image = image[0:orig_shape[0], 0:orig_shape[1]]

                # Image.fromarray(np.uint8((image>0.25)*255)).save('/Users/gm515/Desktop/pred/'+str(slice_number)+'.tif')

                # Remove objects smaller than chosen size
                image = label(image > 0.5, connectivity=image.ndim)  # 0.25

                # Get centroids list as (row, col) or (y, x)
                centroids = [region.centroid for region in regionprops(image) if ((region.area > size) and (region.area < 10*size) and (((4 * math.pi * region.area) / (region.perimeter * region.perimeter)) > 0.7))]

                # Add 1 to slice number to convert slice in index to slice file number
                if row_idx is not None:
                    # Convert coordinate of centroid to coordinate of whole image if mask was used
                    if hemseg_image is not None:
                        coordfunc = lambda celly, cellx: (col_idx[cellx], row_idx[celly], slice_number, int(hemseg_image[celly,cellx]))
                    else:
                        coordfunc = lambda celly, cellx: (col_idx[cellx], row_idx[celly], slice_number)
                else:
                    coordfunc = lambda celly, cellx: (cellx, celly, slice_number)

                # Centroids are currently (row, col) or (y, x)
                # Flip order so (x, y) using coordfunc
                centroids = [coordfunc(int(c[0]), int(c[1])) for c in centroids]

            # Write out results to file
            csv_file = os.path.join(count_path, outdir, str(name)+'_unet_count_INQUEUE.csv')

            # Write out detected centroids to CSV
            # File is locked until writing is complete to prevent writing centroids out of order
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
                    if e.errno != e.errno.EAGAIN:
                        raise
                    else:
                        time.sleep(0.1)

            print('Finished - Queue position: '+str(slice_number)+' Structure: '+str(name))
