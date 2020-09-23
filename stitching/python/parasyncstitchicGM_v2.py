"""
################################################################################
Parallel-and-asynchronous Stitching Script - Parallel version

Author: Gerald M

This script pulls the data generated through TissueCyte (or another microscope system) and
can perfom image averaging correction on the images if requested, before calling ImageJ
from the command line to perform the stitching. You will need to have the plugin script
OverlapY.ijm installed in ImageJ in order for the difference in the X and Y overlap to be
registered. Otherwise the X overlap will be used for both.

The pipeline has been sped up in some areas by parallelising some functions.

Installation:
1) Navigate to the folder containing the parasyncstitchGM.py
2) Run 'pip install -r requirements.txt'

Instructions:
1) Run the script in a Python IDE (e.g. for Python 3 > exec(open('parasyncstitchicGM_v2.py').read()))
2) Fill in the parameters that you are asked for
   Note: You can drag and drop folder paths (works on MacOS) or copy and paste the paths
   Note: The temporary directory is required to speed up ImageJ loading of the files

Important updates:
06.03.19 -  Updated the overlap and crop parameters to improve the image average result and
            tiling artefacts.
11.03.19 -  Included default values and parameter search from Mosaic file.
02.05.19 -  Python 3 compatible
14.05.19 -  Added slack integration
31.01.20 -  Changed option for 16-bit output with improved scaling
05.02.20 -  Removed 16-bit
05.02.20 -  Added feathering to tile edge and stitching with max intensity
10.02.20 -  Added average tile from flatfield imaging and uses that for correction
################################################################################
"""

import cv2, os, sys, warnings, time, glob, errno, subprocess, shutil, readline, re, tempfile, random
import numpy as np
import tifffile
from PIL import Image
from multiprocessing import Pool, cpu_count, Array, Manager
from functools import partial
from datetime import date

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

#=============================================================================================
# Slack notification
#=============================================================================================

def slack_message(text, channel, username):
    from urllib import request, parse
    import json

    post = {
            "text": "{0}".format(text),
            "channel": "{0}".format(channel),
            "username": "{0}".format(username),
            "icon_url": "https://github.com/gm515/gm515.github.io/blob/master/Images/imperialstplogo.png?raw=true"
            }

    try:
        json_data = json.dumps(post)
        req = request.Request('https://hooks.slack.com/services/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',
            data=json_data.encode('ascii'),
            headers={'Content-Type': 'application/json'})
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))

#=============================================================================================
# Function to load images in parallel
#=============================================================================================
def load_tile(file, cropstart, cropend):
    if '_00' in file:
        try:
            tileimage_ch1 = np.array(Image.fromarray(tifffile.imread(file.replace('_00', '_01'))).crop((cropstart, cropstart+65, cropend, cropend+65)).rotate(90))
            tileimage_ch2 = np.array(Image.fromarray(tifffile.imread(file.replace('_00', '_01'))).crop((cropstart, cropstart+65, cropend, cropend+65)).rotate(90))
            tileimage_ch3 = np.array(Image.fromarray(tifffile.imread(file.replace('_00', '_01'))).crop((cropstart, cropstart+65, cropend, cropend+65)).rotate(90))
            tileimage = np.maximum(tileimage_ch1, tileimage_ch2)
            tileimage = np.maximum(tileimage, tileimage_ch3)
        except (ValueError, IOError, OSError):
            tileimage = np.zeros((cropend-cropstart, cropend-cropstart))
    else:
        try:
            tileimage = np.array(Image.fromarray(tifffile.imread(file)).crop((cropstart, cropstart+65, cropend, cropend+65)).rotate(90))
        except (ValueError, IOError, OSError):
            tileimage = np.zeros((cropend-cropstart, cropend-cropstart))
    return tileimage

#=============================================================================================
# Function to check operating system
#=============================================================================================
# This check is to determine which file paths to use if run on the local Mac or Linux supercomputer
def get_platform():
    platforms = {
                 'linux1' : 'Linux',
                 'linux2' : 'Linux',
                 'darwin' : 'Mac',
                 'win32' : 'Windows'
                 }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

#=============================================================================================
# Generate an intensity correction tile
#=============================================================================================
def generate_corr(tcpath, scanid, startsec, endsec):
    # Generate all possible folders in the scan directory
    folderlist = []
    for section in range(startsec,endsec+1,1):
        if section <= 9:
            sectiontoken = '000'+str(section)
        elif section <= 99:
            sectiontoken = '00'+str(section)
        else:
            sectiontoken = '0'+str(section)

        folderlist.append(scanid+'-'+sectiontoken)

    # Generate all possible tile paths for the channel in scan directory
    tilelist = []
    for folder in folderlist:
        tilelist += glob.glob(os.path.join(tcpath, folder, '*_0'+channel+'.tif'))

    average_num = 500
    if len(tilelist) < average_num:
        average_num = len(tilelist)

    print ('Generating correction tile from n='+str(average_num)+' of '+str(len(tilelist))+' total tiles.')

    tilelist = random.sample(tilelist, average_num)

    size = Image.open(tilelist[0]).size
    cropstart = int(round(0.014*size[0])) #0.0096 but leaves a bit too much on edge
    cropend = int(round(size[0]-cropstart+1))

    img_arr = np.array([np.array(Image.open(file).crop((cropstart, cropstart, cropend, cropend)).rotate(90)) for file in tilelist])
    img_arr = img_arr.astype(np.float32)

    # Average image
    img_arr_nan = img_arr
    img_arr_nan[img_arr_nan==0] = np.nan
    avgimage = np.nanmean(img_arr_nan, axis=0)

    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/0-average-tile.tif')

    # Blur the result to remove random artefacts
    avgimage = cv2.GaussianBlur(avgimage, (1501,1501), 0)

    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/1-gaussian-tile.tif')

    padsize = 144
    weight_func = lambda x : np.sqrt(padsize**2 - x**2)/padsize
    weight = np.flip(np.array([weight_func(x) for x in range(0, padsize+1)]))

    # Normalise the average image so all we need to do is multiply each tile by the correction (returned)
    avgimage = avgimage/np.max(avgimage)
    avgimage = 2-avgimage

    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/2-correction-tile.tif')

    # Scale the pad area by the weight
    copy = avgimage
    for pos in np.flip(range(padsize+1)):
        avgimage[pos,:] = copy[pos,:]*weight[pos]
        avgimage[-pos,:] = copy[-pos,:]*weight[pos]
        avgimage[:,pos] = copy[:,pos]*weight[pos]
        avgimage[:,-pos] = copy[:,-pos]*weight[pos]

    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/3-feathered-tile.tif')

    return avgimage

#=============================================================================================
# Generate an intensity correction tile version 2 using flat field
#=============================================================================================
def generate_corr_v2(avg_tile_path):
    avgimage = Image.open(avg_tile_path)

    size = avgimage.size
    cropstart = int(round(0.014*size[0])) #0.0096 but leaves a bit too much on edge
    cropend = int(round(size[0]-cropstart+1))

    avgimage = np.array(avgimage.crop((cropstart, cropstart, cropend, cropend)).rotate(90)).astype(np.float32)
    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/0-average-tile.tif')

    avgimage = cv2.GaussianBlur(avgimage, (201,201), 0)
    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/1-gaussian-tile.tif')

    avgimage = avgimage/np.max(avgimage)
    avgimage = 2-avgimage
    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/2-correction-tile.tif')

    padsize = 140
    weight_func = lambda x : np.sqrt(padsize**2 - x**2)/padsize
    weight = np.flip(np.array([weight_func(x) for x in range(0, padsize+1)]))

    copy = avgimage
    for pos in np.flip(range(padsize+1)):
        avgimage[pos,:] = copy[pos,:]*weight[pos]
        avgimage[-pos,:] = copy[-pos,:]*weight[pos]
        avgimage[:,pos] = copy[:,pos]*weight[pos]
        avgimage[:,-pos] = copy[:,-pos]*weight[pos]

    # Image.fromarray((255*avgimage/np.max(avgimage)).astype(np.uint8)).save('/Users/gm515/Desktop/3-feathered-tile.tif')

    return avgimage


if __name__ == '__main__':
    #=============================================================================================
    # Function to check operating system
    #=============================================================================================
    if get_platform() == 'Mac':
        imagejpath = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
        overlapypath = '"/Applications/Fiji.app/plugins/OverlapY.ijm"'
    if get_platform() == 'linux':
        imagejpath = '/opt/fiji/Fiji.app/ImageJ-linux64'
        overlapypath = '"/opt/fiji/Fiji.app/plugins/OverlapY.ijm"'
    if get_platform() == 'Windows':
        imagejpath = 'fill in path to imagej executable'
        overlapypath = '"fill in path to OverlapY.ijm"'

    #=============================================================================================
    # Input parameters
    #=============================================================================================
    print ('')
    print ('------------------------------------------')
    print ('             Parameter Input              ')
    print ('------------------------------------------')
    print ('')
    print ('Fill in the following variables. To accept default value, leave response blank.')
    print ('Please note this creates a temporary folder to hold images. You require at least 1 GB of free space.')
    acknowledge = input('Press Enter to continue: ')
    tcpath = input('Select TC data directory (drag-and-drop or type manually): ').rstrip()
    startsec = input('Start section (default start): ')
    endsec = input('End section (default end): ')
    xoverlap = input('X overlap % (default 7.2): ')
    yoverlap = input('Y overlap % (default 7.2): ')
    channel = input('Channel to stitch (0 - combine all channels): ')
    while not channel:
        channel = input('Channel to stitch (0 - combine all channels): ')
    avgcorr = input('Perform average correction? (y/n): ')
    while avgcorr not in ('y', 'n'):
        avgcorr = input('Perform average correction? (y/n): ')
    convert = input('Perform additional downsize? (y/n): ')
    while convert not in ('y', 'n'):
        convert = input('Perform additional downsize? (y/n): ')
    if convert == 'y':
        downsize = input('Downsize amount (default 0.054 for 10 um/pixel): ')

    # Handle default values
    if not startsec:
        startsec = 1
    else:
         startsec = int(startsec)
    if endsec:
        endsec = int(endsec)
    if not xoverlap:
        xoverlap = 7.2
    else:
        xoverlap = float(xoverlap)
    if not yoverlap:
        yoverlap = 7.2
    else:
        yoverlap = float(yoverlap)
    if convert == 'y':
        if not downsize:
            downsize = 0.054
        else:
            downsize = float(downsize)

    # Search the mosaic file for remaining parameters
    mosaicfile = glob.glob(os.path.join(tcpath, 'Mosaic*.txt'))[0]
    with open(mosaicfile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'sections' in line:
                trueendsec = int(re.split(':', line.rstrip())[-1])
            if 'Sample ID' in line:
                scanid = re.split(':', line.rstrip())[-1]
            if 'mrows' in line:
                xtiles = int(re.split(':', line.rstrip())[-1])
            if 'mcolumns' in line:
                ytiles = int(re.split(':', line.rstrip())[-1])
            if 'layers' in line:
                zlayers = int(re.split(':', line.rstrip())[-1])

    if not endsec:
        endsec = trueendsec

    # Create stitch output folders
    os.umask(0o000)

    stitchpath = os.path.join(tcpath, scanid+'-Mosaic', 'Ch'+str(channel)+'_Stitched_Sections_'+str(date.today().strftime('%d_%m_%Y')))

    try:
        os.makedirs(stitchpath, 0o777)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if convert == 'y':
        try:
            os.makedirs(stitchpath+'_Downsized', 0o777)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # create temporary folder in home path
    temppath = tempfile.mkdtemp(dir=os.path.expanduser('~'))
    tilepath = temppath

    crop = 0
    filenamestruct = []
    tstart = time.time()
    zcount = ((startsec-1)*zlayers)+1
    filenumber = 0
    tilenumber = 0
    lasttile = -1
    tileimage = 0

    corrtile = None

    #=============================================================================================
    # Stitching
    #=============================================================================================
    print ('')
    print ('---------------------------------------------')
    print ('          Parasynchronous Stitching          ')
    print ('---------------------------------------------')
    print ('')

    tstart = time.time()

    # Check that data exists
    for section in range(startsec,endsec+1,1):
        if section <= 9:
            sectiontoken = '000'+str(section)
        elif section <= 99:
            sectiontoken = '00'+str(section)
        else:
            sectiontoken = '0'+str(section)

        folder = scanid+'-'+sectiontoken

        # Token variable hold
        x = xtiles
        y = ytiles
        xstep = -1

        for layer in range(1,zlayers+1,1):
            completelayer = False
            firsttile = xtiles*ytiles*((zlayers*(section-1))+layer-1)
            lasttile = (xtiles*ytiles*((zlayers*(section-1))+layer))-1

            # If last tile doesn't exist yet, wait for it
            if glob.glob(os.path.join(tcpath, folder, '*-'+str(lasttile)+'_0*.tif')) == []:
                while glob.glob(os.path.join(tcpath, folder, '*-'+str(lasttile)+'_0*.tif')) == []:
                    sys.stdout.write('\rLast tile not generated yet. Waiting.')
                    sys.stdout.flush()
                    time.sleep(3)
                    sys.stdout.write('\rLast tile not generated yet. Waiting..')
                    sys.stdout.flush()
                    time.sleep(3)
                    sys.stdout.write('\rLast tile not generated yet. Waiting...')
                    sys.stdout.flush()

            time.sleep(3)
            # Get file name structure and remove last 8 characters to leave behind filename template
            filenamestruct = glob.glob(os.path.join(tcpath, folder, '*-'+str(lasttile)+'_01.tif'))[0].rpartition('-')[0]+'-'
            filenames = [filenamestruct+str(tile)+'_0'+str(channel)+'.tif' for tile in range(firsttile, lasttile+1, 1)]

            # Get crop value
            if crop == 0:
                onefile = filenames[-1]
                if channel == '0':
                    onefile = onefile.replace('_00', '_01')
                size = Image.open(onefile).size[0]
                cropstart = int(round(0.014*size)) #0.0075, originally 0.045
                cropend = int(round(size-cropstart+1))
            crop = 1

            # Load images through parallel
            pool = Pool(cpu_count())
            img_arr = np.squeeze(np.array([pool.map(partial(load_tile, cropstart=cropstart, cropend=cropend), filenames)]), axis=0).astype(np.float32)
            pool.close()
            pool.join()

            # Compute average tile and divide if required
            if avgcorr == 'y':
                if corrtile is None:
                    # corrtile = generate_corr(tcpath, scanid, startsec, trueendsec)
                    avg_tile_path = '../AVG_IMG.tif'
                    corrtile = generate_corr_v2(avg_tile_path)
                    print ('Computed correction tile from 1000 random samples.')

                # Correct each image
                img_arr = np.multiply(img_arr, corrtile)

            sectionmax = np.max(img_arr)

            # Save each tile with corresponding name
            for tile_img in img_arr:
                if x>=1 and x<=xtiles:
                    if x<10:
                        xtoken = '00'+str(x)
                    else:
                        xtoken = '0'+str(x)
                    x+=xstep
                    if y<10:
                        ytoken = '00'+str(y)
                    else:
                        ytoken = '0'+str(y)
                elif x>xtiles:
                    x = xtiles
                    if x<10:
                        xtoken = '00'+str(x)
                    else:
                        xtoken = '0'+str(x)
                    xstep*=-1
                    x+=xstep
                    y+=-1
                    if y<10:
                        ytoken = '00'+str(y)
                    else:
                        ytoken = '0'+str(y)
                elif x<1:
                    x=1
                    if x<10:
                        xtoken = '00'+str(x)
                    else:
                        xtoken = '0'+str(x)
                    xstep*=-1
                    x+=xstep
                    y+=-1
                    if y<10:
                        ytoken = '00'+str(y)
                    else:
                        ytoken = '0'+str(y)

                if zcount < 10:
                    ztoken = '00'+str(zcount)
                elif zcount < 100:
                    ztoken = '0'+str(zcount)
                else:
                    ztoken = str(zcount)

                # Was divide by 1000.
                tile_img = np.multiply(np.divide(tile_img, sectionmax), 255.).astype(np.uint8)

                # Image.fromarray(tile_img).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(tilepath, 'Tile_Z'+ztoken+'_Y'+ytoken+'_X'+xtoken+'.tif'))
                Image.fromarray(tile_img).save(os.path.join(tilepath, 'Tile_Z'+ztoken+'_Y'+ytoken+'_X'+xtoken+'.tif'))

            print ('Stitching Z'+ztoken+'...')

            outname = "'"+os.path.join(stitchpath, 'Stitched_Z'+ztoken+'.tif')+"'"

            subprocess.call([imagejpath, '--headless', '-eval', 'runMacro('+overlapypath+');', '-eval', 'run("Grid/Collection stitching", "type=[Filename defined position] grid_size_x='+str(xtiles)+' grid_size_y='+str(ytiles)+' tile_overlap_x='+str(xoverlap)+' tile_overlap_y='+str(yoverlap)+' first_file_index_x=1 first_file_index_y=1 directory=['+tilepath+'] file_names=Tile_Z'+ztoken+'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_Z'+ztoken+'.txt fusion_method=[Max. Intensity] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");', '-eval', 'saveAs("Tiff",'+outname+');'], stdout=open(os.devnull, 'wb'))

            shutil.rmtree(temppath)
            os.makedirs(temppath, 0o777)

            if convert == 'y':
                stitched_img = np.array(Image.open(os.path.join(stitchpath, 'Stitched_Z'+ztoken+'.tif')))
                stitched_img = Image.fromarray(np.multiply(np.divide(stitched_img.astype(float),np.max(stitched_img.astype(float))), 255.).astype('uint8'), 'L')
                stitched_img = stitched_img.resize((int(downsize*stitched_img.size[0]), int(downsize*stitched_img.size[1])), Image.ANTIALIAS)
                stitched_img.save(os.path.join(stitchpath+'_Downsized', 'Stitched_Z'+ztoken+'.tif'))

            print ('Complete!')

            zcount+=1
            y = ytiles
            x = xtiles
            xstep = -1

            tilenumber+=1

    #=============================================================================================
    # Finish
    #=============================================================================================
    shutil.rmtree(temppath)

    minutes, seconds = divmod(time.time()-tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    text = 'Stitching completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)

    print ('')
    print (text)
    slack_message(text, '#stitching', 'Stitching')
