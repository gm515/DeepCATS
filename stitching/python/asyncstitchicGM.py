#============================================================================================
# Asynchronous Stitching Script
# Author: Gerald M
#
# This script pulls the data generated through TissueCyte (or another microscope system) and
# can perfom image averaging correction on the images if requested, before calling ImageJ
# from the command line to perform the stitching. You will need to have the plugin script
# OverlapY.ijm installed in ImageJ in order for the difference in the X and Y overlap to be
# registered. Otherwise the X overlap will be used for both.
#
# Installation:
# 1) Navigate to the folder containing the AsyncstitchGM.py
# 2) Run 'pip install -r requirements.txt'
#
# Instructions:
# 1) Run the script in a Python IDE
# 2) Fill in the parameters that you are asked for
#    Note: You can drag and drop folder paths (works on MacOS) or copy and paste the paths
#    Note: The temporary directory is required to speed up ImageJ loading of the files
#============================================================================================

import os, sys, warnings, time, glob, errno, subprocess, shutil, math
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

#=============================================================================================
# Check operating system
#=============================================================================================
# This check is to determine which file paths to use if run on the local Mac or Linux supercomputer
def get_platform():
    platforms = {
        'linux1' : 'Linux',
        'linux2' : 'Linux',
        'darwin' : 'Mac'
    }
    if sys.platform not in platforms:
        return sys.platform

    return platforms[sys.platform]

if get_platform() == 'Mac':
    imagejpath = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
    overlapypath = '"/Applications/Fiji.app/plugins/OverlapY.ijm"'
if get_platform() == 'Linux':
    imagejpath = '/home/gm515/Fiji.app/ImageJ-linux64'
    overlapypath = '"/home/gm515/Fiji.app/plugins/OverlapY.ijm"'

#=============================================================================================
# Input parameters
#=============================================================================================

tcpath = raw_input('Select TC data directory (drag-and-drop): ').rstrip()
temppath = raw_input('Select temporary directory (drag-and-drop): ').rstrip()

scanid = raw_input('Scan ID: ')
startsec = input('Start section: ')
endsec = input('End section: ')
xtiles = input('Number of X tiles: ')
ytiles = input('Number of Y tiles: ')
zlayers = input('Number of Z layers per slice: ')
xoverlap = input('X overlap % (default 5): ')
yoverlap = input('Y overlap % (default 6): ')
channel = input('Channel to stitch: ')
avgcorr = raw_input('Perform average correction? (y/n): ')
convert = raw_input('Perform additional downsize? (y/n): ')
if convert == 'y':
    downsize = input('Downsize amount (default 0.054 for 10 um/pixel): ')

# Create folders
os.umask(0000)
try:
    os.makedirs(tcpath+'/'+str(scanid)+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections', 0777)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if convert == 'y':
    try:
        os.makedirs(tcpath+'/'+str(scanid)+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections_Scaled', 0777)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Check if temporaty folder is empty
if os.listdir(temppath) != []:
    raise Exception('Temporary folder needs to be empty!')

crop = 0
filenamestruct = []
tstart = time.time()
zcount = ((startsec-1)*zlayers)+1
filenumber = 0
tilenumber = 0
lasttile = -1
tileimage = 0

#=============================================================================================
# Stitching
#=============================================================================================

print ''
print '------------------------------------------'
print '          Asynchronous stitching          '
print '------------------------------------------'

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
        if glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0*.tif') == []:
            while glob.glob(tcpath+'/'+folder+'/*-'+str(lasttile)+'_0*.tif') == []:
                sys.stdout.write('\rLast tile not generated yet. Waiting.')
                sys.stdout.flush()
                time.sleep(3)
                sys.stdout.write('\rLast tile not generated yet. Waiting..')
                sys.stdout.flush()
                time.sleep(3)
                sys.stdout.write('\rLast tile not generated yet. Waiting...')
                sys.stdout.flush()

        filenumber = firsttile
        # Get file name structure and remove last 8 characters to leave behind filename template
        filenamestruct = glob.glob(tcpath+'/'+folder+'/*-'+str(filenumber)+'_0'+str(channel)+'.tif')[0].rpartition('-')[0]+'-'

        for tile in range(firsttile, lasttile+1, 1):
            # Try to open file. If it doesn't exist, create an empty file
            try:
                tileimage = Image.open('/'+filenamestruct+str(filenumber)+'_0'+str(channel)+'.tif')
            except IOError as e:
                if e.errno == errno.ENOENT:
                    tileimage = Image.fromarray(np.zeros(tileimage.size))
                else:
                    raise

            # Set crop value if not already stored
            if crop == 0:
                crop = round(0.018*tileimage.size[0])

            # Crop and rotate image and convert to numpy array
            tileimage2 = np.array(tileimage.crop((crop, crop, tileimage.size[0]-crop+1, tileimage.size[1]-crop+1)).rotate(90))

            if tile == firsttile:
                sumimage = tileimage2
            else:
                sumimage = sumimage + tileimage2

            filenumber+=1

        # Compute average tile
        avgimage = sumimage.astype(float)/(xtiles*ytiles)
        print 'Computed average tile.',

        tilenumber = firsttile

        for tile in range(firsttile, lasttile+1, 1):
            # Try to open file. If it doesn't exist, create an empty file
            try:
                tileimage = Image.open('/'+filenamestruct+str(tilenumber)+'_0'+str(channel)+'.tif')
            except IOError as e:
                if e.errno == errno.ENOENT:
                    tileimage = Image.fromarray(np.zeros(tileimage.size))
                else:
                    raise

            tileimage2 = np.array(tileimage.crop((crop, crop, tileimage.size[0]-crop+1, tileimage.size[1]-crop+1)).rotate(90)).astype(float)

            if avgcorr == 'y':
                tileimage2 = np.multiply(np.divide(tileimage2, avgimage, where=avgimage!=0.), 1000)


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

            Image.fromarray(tileimage2.astype(np.uint16)).save(temppath+'/Tile_Z'+ztoken+'_Y'+ytoken+'_X'+xtoken+'.tif')

            if (tile+1)%(xtiles*ytiles) == 0:
                print 'Stitching Z'+ztoken+'...',

                tilepath = temppath+'/'
                stitchpath = tcpath+'/'+scanid+'-Mosaic/Ch'+str(channel)+'_Stitched_Sections'
                subprocess.call([imagejpath, '--headless', '-eval', 'runMacro('+overlapypath+');', '-eval', 'run("Grid/Collection stitching", "type=[Filename defined position] grid_size_x='+str(xtiles)+' grid_size_y='+str(ytiles)+' tile_overlap_x='+str(xoverlap)+' tile_overlap_y='+str(yoverlap)+' first_file_index_x=1 first_file_index_y=1 directory=['+tilepath+'] file_names=Tile_Z'+ztoken+'_Y{yyy}_X{xxx}.tif output_textfile_name=TileConfiguration_Z'+ztoken+'.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=['+stitchpath+']");'], stdout=open(os.devnull, 'wb'))

                shutil.rmtree(temppath)
                os.makedirs(temppath, 0777)

                os.rename(stitchpath+'/img_t1_z1_c1', stitchpath+'/Stitched_Z'+ztoken+'.tif')

                if convert == 'y':
                    stitched_img = np.array(Image.open(stitchpath+'/Stitched_Z'+ztoken+'.tif')).astype(float)
                    stitched_img = Image.fromarray(np.multiply(np.divide(stitched_img,65535.), 255.).astype(np.uint8))
                    stitched_img = stitched_img.resize((int(downsize*stitched_img.size[0]), int(downsize*stitched_img.size[1])))
                    stitched_img.convert('L').save(stitchpath+'_JPEG/Stitched_Z'+ztoken+'.jpg')

                print 'Complete!'

                zcount+=1
                y = ytiles
                x = xtiles
                xstep = -1

            tilenumber+=1

#=============================================================================================
# Finish
#=============================================================================================

minutes, seconds = divmod(time.time()-tstart, 60)
hours, minutes = divmod(minutes, 60)
days, hours = divmod(hours, 24)

print ''
print 'Stitching completed in %02d:%02d:%02d:%02d' %(days, hours, minutes, seconds)
