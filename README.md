# DeepCATS

Repository for **Deep**learning **C**ounting in **A**natomically **T**argeted **S**tructures and general TissueCyte usage. The goal of this repo is to provide the tools and instructions for stitching TissueCyte images, registering with the Allen Brain Atlas and cell counting using the DeepCATS pipeline. 

## Stitching

During the entire processing pipeline, the first thing to do is stitch the tile images generated through TissueCyte. Stitching can be conducted alongside the acquisition, so it is recommended that as soon as the TissueCyte scan is initiated, you go to the computer which will process the data and set up the stitching.

### Installation

Get set up with Anaconda (Python 3) if it is not already set up, create an environment, and make sure you have the following requirement packages installed in the environment. These can be installed with `pip` or through `conda`.

```
cv2 (opencv)
glob2
numpy
Pillow or Pillow-SIMD
tifffile
```

Note that for `Pillow` I would recommend installing `Pillow-SIMD` from https://github.com/uploadcare/pillow-simd which has faster read speeds and is generally faster for other Pillow related image processing functions. Installation of `Pillow-SIMD` can create issues during compiling. If so, see here https://github.com/uploadcare/pillow-simd/issues/9#issuecomment-249357973 for the following installation command,

`pip install --upgrade pillow-simd --global-option="build_ext" --global-option="--disable-jpeg" --global-option="--disable-zlib"`

The stitching also requires ImageJ/Fiji and the grid/collection stitching plugin. Head over to https://fiji.sc and download a copy of Fiji. When it is installed, load it up and go `Plugins > Stitching` and check that `Grid/Collection stitching` exists. 

By default, when stitching the plugin only handles a single overlap value between tiles which is configured for both the X and Y overlap. However, the plugin has a weird hidden setting which can only be triggered through a BeanShell script that allows seperate X and Y overlap values to be used; hence the existence of a file called `OverlapY.ijm`. We need to tell ImageJ/Fiji to execute this prior to running the stitching so we can parse in two separate overlap values. We also need to tell the Python script where to find the ImageJ application.

To do this, open the desired stitching script such as `parasyncstitchicGM_v2.py` in a script editor such as Spyder which comes with Anaconda, or any other code IDE (I like using Atom, for example, as it integrates in GitHub quite nicely). `parasyncstitchicGM_v2.py` is probably the best script to use. The other scripts are similar but execute slightly differently from circumventing image loading issues and intensity correction issues. Around lines 217 you will see the following code block. Edit the `imagejpath` to the path for your system. It will likely be similar to what is already listed but it needs to be exact. Also edit the `overlapypath` to match the location of the `OverlapY.ijm` file that resides in the stitching folder. 

```
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
```

Everything should now be set up! To check, run the file with `exec(open("./parasyncstitchicGM_v2.py").read())` and you should be greated with the following

```
------------------------------------------
             Parameter Input
------------------------------------------

Fill in the following variables. To accept default value, leave response blank.
Please note this creates a temporary folder to hold images. You require at least 1 GB of free space.
Press Enter to continue:
```

Everything is correct. If it complains about any missing modules, then go ahead and install them manually using `pip` or `conda`. Just search for `install module`  in Google and you will immediately find how to install it. Ideally you only use either `pip` or `conda` as the installation manager to prevent path conflictions, but realistically conda module versions are not always up to date so you are often forced to use one or the other installation managers for different modules. If you have path conflicts for modules when using Python, safe assured it comes down to using `pip` or `conda`.

### Instructions

1. Make sure the NAS drive or location where the tile data is stored is accessible from the computer doing the stitching. 
2. Activate the environment and run the stitching script `exec(open("./parasyncstitchicGM_v2.py").read())`.
3. You will be greated with the following
```
------------------------------------------
             Parameter Input
------------------------------------------

Fill in the following variables. To accept default value, leave response blank.
Please note this creates a temporary folder to hold images. You require at least 1 GB of free space.
Press Enter to continue:
```
Press enter to confirm.
4. This will be followed by input respones. Fill in as necessary for all the required values. An example is as follows.
```
------------------------------------------
             Parameter Input
------------------------------------------

Fill in the following variables. To accept default value, leave response blank.
Please note this creates a temporary folder to hold images. You require at least 1 GB of free space.
Press Enter to continue:
Select TC data directory (drag-and-drop or type manually): /mnt/TissueCyte80TB/180812_Het
Start section (default start):
End section (default end):
X overlap % (default 7.2):
Y overlap % (default 7.2):
Channel to stitch (0 - combine all channels): 0
Perform average correction? (y/n): y
Perform additional downsize? (y/n): y

Downsize amount (default 0.054 for 10 um/pixel):
```
Note that default values listed can be accessed by just pressing enter and leaving the input value blank. For three channels on TissueCyte, you can use either channels 1-3 for stitching. Option 0 is sometimes handy just to merge everything together across all channels so you see all data. If you are only insterested in GFP signal, for example, you can of course just choose to stitch channel 2. 

If everything is correct, then the stitching should proceed. It may take a while to give you updates on the stitching process. If there is no immediate error and exiting of the script, then everything is likely fine.

This script had to be edited several times so by all means make your own changes to the script. I'd like to think that everything is commented nicely. At one point there was a slack notification integration so you would get notified when the stitching finished. This requires a URL hook for the slack channel you want to ping a notification to, but at one point Slack changed the security to affect how the integration worked, and therefore it doesn't work in its current state. 




## Installation

Get set up with Anaconda (Python 3) then install the deeplearnenv environment with

`conda env create -f deeplearnenv.yml`

The repo contains everything which is needed (plus additional surplus code at the moment).

## Running

To run the automated counting script, first start the flask/gunicorn server by navigating to `flaskgunicornserver` and running the following in the command line.

`gunicorn --timeout 240 --bind 0.0.0.0:5000 wsgi:app`

This will start the server which will perform the classification using a U-Net model loaded in the `flaskunetserver.py` file. This model path can be changed to load in any model.

Next, in a new command line terminal, run the following command to start cell counting in structures of interest.

`ipython deepcats.py -- <*Ch2_Stitched_Sections> -maskpath <*SEGRES.tif> -hempath <*HEMRES.tif> -radius 10 -ncpu 8 -structures <DORsm,DORpm>`

- `<*Ch2_Stitched_Sections>` The path to the microscopy images for counting
- `<*SEGRES.tif>` The path to the annotation atlas registered to the above data set
- `<*HEMRES.tif>` The path to the hemisphere atlas registered to the above data set
- `<DORsm,DORpm>` The list of structures for counting as a list. Avoid inputting too many structures as all images are stored in RAM is limited by the workstation being used.

Additionally, the expected radius of a cell `-radius` and the number of cpus `-ncpu` can be modified by changing the appropriate value. Plus any other parameters listed in `deepcats.py`, or as listed below.

- `imagepath` Image directory path for counting
- `-maskpath` Annotation file path for masking
- `-hempath` Hemisphere file path for hemisphere classification
- `-structures` List of structures to count within
- `-oversample` Oversample correction
- `-start` Start image number if required
- `-end` End image number if required
- `-medfilt` Use custom median donut filter
- `-circthresh` Circularity threshold value
- `-xyvox` XY voxel size
- `-zvox` Z voxel size
- `-ncpu` Number of CPUs to use
- `-size` Approximate radius of detected objects
- `-radius` Approximate radius of detected objects

## Models

Currently the best model to use is 2020_03_18_Adam0.0001_BCE_elu_GPUs4_Batch8_Aug6_GM_UNet with details as follows:

```
============================================

         Model: GM_UNet
Augmentation n: 6 w/ training 906, val 302
 Batch per GPU: 8
        N GPUs: 1
     Optimiser: Adam
          Loss: BCE
    Activation: elu
Best val. loss: 0.1534
Execution time: ?

============================================
```
