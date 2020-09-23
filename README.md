# DeepCATS

Repository for **Deep**learning **C**ounting in **A**natomically **T**argeted **S**tructures and general TissueCyte usage. The goal of this repo is to provide the tools and instructions for stitching TissueCyte images, registering with the Allen Brain Atlas and cell counting using the DeepCATS pipeline. 

This repo has broken down into different approproately-named folders.

- `stitching`: contains anything regarding the tile stitching process
- `registration`: contains anything regarding the registration to the CCF3 atlases
- `deepcats`: contains anything to automatically cell count using the registration result
- `trainingdeepcats`: contains anything required to train the neural network model which does inference in the cell counting pipeline

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

Note that default values listed can be accessed by just pressing enter and leaving the input value blank. For three channels on TissueCyte, you can use either channels 1-3 for stitching. Option 0 is sometimes handy just to merge everything together across all channels so you see all data. If you are only insterested in GFP signal, for example, you can of course just choose to stitch channel 2. I would also recommend leaving the default downsizing in place although this only works if you are doing a 2080 x 2080 pixel TissueCyte scan. For this scenario, the downsize option is chosen to make give a 10 um/pixel resolution which is ideal for the following registration steps. For a different scan resoltuion, choose appropriately or ignore the downsize step and do separately. 

If everything is correct, then the stitching should proceed. It may take a while to give you updates on the stitching process. If there is no immediate error and exiting of the script, then everything is likely fine. All stitched data gets saved into it's own `Mosaic` folder within the scan directory that was input. You may change this if this is not required by editing the script. 

This script had to be edited several times so by all means make your own changes to the script. I'd like to think that everything is commented nicely. At one point there was a slack notification integration so you would get notified when the stitching finished. This requires a URL hook for the slack channel you want to ping a notification to, but at one point Slack changed the security to affect how the integration worked, and therefore it doesn't work in its current state. 

## Registration

When all the data is stitched you will need to downsize to match a 10 um/pixel resolution which is the same resolution as the Allen Institute Common Coordinate Framework V3 (CCF3). This is done as part of the stitching if chosen. You then need to convert all the images into a single TIFF file. Use ImageJ/Fiji for this as it should only be a 1.5GB in total (you are using the downsized images not the original resolution!). You may need to also make some intensity adjustments to the images if it's looking a little too dark. If there is a lot of noise, you may also want to throw a 3D filtering process such as the median filter. The 3D kernel ensures the filter operates along the Z-axis as well as the X and Y. At this point though, you should have a TIFF file which contains the downsampled version of your brain for registration. This is the fixed image data containing the autofluorescence (or as much strucutral information as possible).

### Installation

The registration is done using SimpleElastix. Check out https://simpleelastix.readthedocs.io for the documents, installation, and a guide. It is well worth reading through it as the author, Kasper Marstal, does a great job of explaining the nuances of registration. If SimpleElastix has not been installed, go ahead and install it, ideally in a new Anaconda envirnment, and check it works. Follow the guide for some guidance on the set up.

You may also need to install `skimage`. 

### Instructions

There is a little set up which bascailly requires you have the relevant files for stitching. Make sure you have a folder called `atlases` which contains `annotation_10um.tif` and `average_10um.tif`. These are the annotation files and average files for the CCF3 atlases respectively. You will also need a folder called `parametermaps` containing the parameter maps which will guide the registration process. You need three registration maps for the rigid, affine and bspline components. You also only need my own paramter maps, with `GM` in the name, although there should also be a bunch of others from other literature. The `GM` maps are the most robust however. Feel free to check into the actual parameters in the files themselves as this is handy for getting and idea about what is happening. Some scenarios might require the use of different parameters as well. 

1. Activate your environment which has SimpleElastix and skimage installed. 
2. Run `registration.py` from the command line using the following command and the file path to the downsized data you created earlier. 
`ipython registration.py -- /path/to/downsized/tissuecyte/autofluoresence/data.tif`

By default this will use the `GM` parameter maps and run three stages of registration on your downsized data. There should be output to the console to inform your of progress. Overall expect the process to take around 30 to 40 mins. If there are any errors then check file paths and dive into the script to see what is wrong. You may also input other parameters for registraiton, as `-parameter value`, for example `ipython registration.py -- /path/to/downsized/tissuecyte/data.tif -first 101`. The other parmeters are listed below.
```
autoflpath: 'File path for autofluorescence atlas'
-avgpath: default='atlases/average_10um.tif': 'File path for average atlas'
-annopath: default='atlases/annotation_10um.tif': 'File path for annotation atlas'
-first: default=0: 'First slice in average atlas'
-last: default=1320: 'Last slice in average atlas'
-fixedpts: default=False: 'Path for fixed points'
-movingpts: default=False: 'path for moving points'
```
The `first` and `last` parameters are only required if you think you need to restrict the extent of the CCF3 atlas during registration. Sometimes the initial global alignment performed with the rigid and the affine stages do not actually manage to work that well. It could be that your data doesn't cover the same extent of the brain. Maybe you only imaged the thalamus but the CCF3 atlases contain the full brain. This would be the time to use those paramters to restrict the rostral-caudal extent of the CCF3 atlas to aid the registration. 

The `fixedpts` and `movingpts` are only useful if you want to input a set of landmarked points between the fixed (your data from TissueCyte) and the moving (the CCF3 atlases) to aid the registration. Each listed point has correspondance between each data set and tells Elastix to reduce the overall distance between corresponding points during the registration. Manual guidance in this way is "handy" but time consuming and is also non-repeatable between samples for the simple fact that you will be landmarking different spots with a different number of landmarks between different samples. Fixed and moving points are however, useful for assessing registration accuracy as you can calculate a Euclidean distance error from the points. I would recomment playing around with the fized and moving points. Have a look at `forward_transform_pts_example` and the `transformpoints.py` script for an example on how points can be used to forward transform during registration to calculate an error.

After the registration is complete, there should be a folder called `Registration_<paramterfilenames>_<date>` with files generated in the folder where the autofluorescence TissueCyte TIFF data (i.e. the file path you input in step 2.). You may want to open the files and overlay them onto the autofluorescence data to check everything matches up. This can be done in ImageJ/Fiji. As a tip, the segmentation result is saved as a 32-bit TIFF file, but the range of pixel intensities is squashed into the first few thousand values from a possible range of 2<sup>32</sup>. This makes the segmentation result look completely empty if you edit the contrast in ImageJ/Fiji and set the upper threshold (use the Set button as otherwise it becomes fiddly), and change the upper limit to 2000. This will compress all pixel values to between 0-2000 and will make things much easier to see. You can also run an edge filter to convert the structures to outlines and then use the colour through `Image > Color Merge > Channels` to overlay the outlines of the segmentation result with the autofluorescence. There should also be a hemisphere atlas created which divides the annotation atlas in half and is needed to do the cell counting in the next step. 

## DeepCATS cell counting using the registration result

The cell counting pipeline uses the registration result to target counting to particular structures. You could also ignore this and simply count in a stack of images. This section only concerns the execution of the pipeline. The next section will look at creating new models, adding data and generally improving the accuracy. The important modules for this are Keras and TensorFlow so it is worth looking up what they are and how to use them!

### Installation

Get set up with Anaconda (Python 3) then install the deeplearnenv environment using the file in the `additionaltools` directory.

`conda env create -f deeplearnenv.yml`

You do not need to create a new environment prior to this as the goal of the above command is to duplicate the envirnment in the `deeplearnenv.yml` file. 

### Instructions

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

### Models

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

## Adding/Improving DeepCATS

### Adding new data

The cell counting pipeline can be modified to use newer models trained on additional data. First, all the training data can be found under the `data` directory. There is a list of different data sets for different uses. The data set used for the model above was `GM_MG_combined_data_n217`. Each data folder is structured in the same way, however. There are `images` and `masks` which contain the image data as 512 x 512 pixel shots taken from serial two-photon scan with TissueCyte, alongside manually drawn masks. Images are 8-bit at 0.54 pixel resolution in XY. Masks are 8-bit with 0.54 pixel resolution in XY, with pixel value 255 denoting cell signal. If you want to add more training data to the model, you are free to create a new folder (or add to the existing folder) with your own data set. This data could be your own manual segmentation or from repositories online. Overall, the more training data, the better!

### Different U-Net architectures

Different neural network architectures can be found in `architectures`. These are the actualy structures of the neural networks. In the cases for the publication and thesis, the neural network architecture was the U-Net created under `unetmodelv2.py`. There exists other versions of the network such as the Nested U-Net and MultiRes U-Net. Versions of these models were found online in other GitHub respositories and were initially tested but did not show any improvement. However, that could be for lack of trying rather than issues with the model architecture themselves. You may find several different versions of the U-Net model online in other GitHub repos, but ulitimately, they all perform differently. Some might use dropout layers, some might use batch normalisation, some might do weights transpose convolution whilst some to simple upscaling transposed convolution. Frustratingly, there is little consistency for the same networks with many different interpretations. For this reason it is also worth exploring your own versions. 

### Training a new model with new data(?), or new architecture(?), or anything(?)

The scripts for training are broken down into five files.

- `augmentation.py`: concerns with the data augmentation to bolster the data set size for training and validation
- `cleanup.py`: concerns simple commands to clean up the directory after training to remove anything not needed
- `losses.py`: contains a bunch of different loss functions from sources online or my own
- `preprocessing.py`: does some preprocessing of the training data to prepare it for use with the network training
- `trainmodel.py`: the script which does everything from preparing to training the model
- `fullimageprediction.py`: example script which shows how to predict with a new image

To start with, if you'd like to explore different loss function, you can explore the `losses.py` script and add your own. I found a bunch online or wrote my own functions. Different loss functions are tailored to different network challanges. Image segmentation is tricky so loss functions for use here typically stick to functions which use true or false, positive or negative detections. Have a look online though as new functions are published all the time. 

Next, the augmentation side can generally be left alone. However, you might want to modify the actual transformations that are done. Look under the `def augment(dir, n):` function to see what transformations and parameters are being conducted. You can comment out some of these lines or add your own or change the parameter values to allow for more drastic or constrained transformations. You can also write a custom class as I did for the Poisson noise to do your own trannsformation which is not provided by the Augmentor module. 

If you would like to modify anything about how the data is preprocessed, then look at the `preprocessing.py` script. Preprocessing includes anything with collating the data together, running the augmentation and splitting the data into the relevant temporary folders (Augmentor dumps everything into the same directory so it is necessary to split things back up again). It also importantly splits the data into training and validation directories with a 75% training and 25% validation split. The script also goes through to identify any NaN or problematic images which might be completely empty or have bad values in them, etc. I found that this was important to ensuring the training went smoothly. Sometimes I made a change to the `augmentation.py` script which resulted in some bad images being generated which would lead to bad training. I wouldn't know about it until it came to actually training, at which point I usually went to the `preprocessing.py` script to figure out why and to fix it. 

Finally, the actual training is executed by running the `trainmodel.py` through the command line as 

`ipython -- trainmodel.py Adam 1e-4 BCE elu 1 GM_UNet 6` 

where each parameter is listed to declare the specifics of training. I listed some obvious go-to parameters under the `if __name__ == '__main__':` function. For example `Adam` uses the Adam optimiser, or `SGD` uses the SGD optimiser. For any others you can add to the script. It just makes executing the model training a little simpler as you can pass in parameters. The same is true for the learning rate `1e-4` as above, the loss function `BCE` as above, the activation function `elu` as above, the number of GPUs `1` as above, the model architecture `GM_UNet` as above, and the amount of augmentation fold to perform `6`-fold as above. Again, you can add any other parameters you would like if you have a new loss function you want to try. Setting the number of GPUs is useful if you will run the training on the hpc cluster. In which case you can get set it to something like `4` and use 4 GPUs at once to really expedite the training. Parameter order is important as the script reads the parameters sequentially (this could be changed so parameters are declared using a `-parmeter value` system). 

An important line is 

`training_dir = 'data/GM_MG_combined_data_n217'`

where you will want to put the location of the new folder of data if you have added your own.

Another important section is
```
    with strategy.scope():
        if model_arg6 == 'GM_UNet': model = unetmodelv2.unet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act) # Works best
        if model_arg6 == 'GM_UNetv3': model = unetmodelv3.unet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act) # Works best
        if model_arg6 == 'ZX_UNet': model = unetmodel.unet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act)
        if model_arg6 == 'ZX_NestedUNet': model = nestedunetmodel.nestedunet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act)
        if model_arg6 == 'ZX_MultiResUNet': model = multiresunetmodel.multiresunet(inputsize=(None, None, 1), optfn=optimizer, lossfn=loss, actfn=act) # Does not work

        model.compile(optimizer=optimizer, loss=[loss], metrics=[losses.dice_loss, loss])
```
where you will need to check that the model architecture is correctly chosen (`GM_UNet` works fine last I checked). Then the last line is necessary for declaring what loss is required. Notice there are two metrics as `metrics=[losses.dice_loss, loss]`. The loss function you choose is the one which is used for training and is set as the variable `loss` earlier in the script. The other is `losses.dice_loss` which is a DICE loss function. This value is used to select the best model for learning as declared in the line
`checkpoint = ModelCheckpoint(filepath, monitor='val_dice_loss', verbose=1, save_best_only=True, mode='min')`
where we explictely state that we want the model with the minimum dice loss (i.e. the model which gives the best performance according to the DICE metric).

Another important line is

`early = EarlyStopping(monitor='val_dice_loss', mode='min', patience=30, verbose=1)`

which evokes early stopping if the validation DICE loss does not decrease after 30 training epochs. This is done to stop unnecessary processing when the model is clearly not learning any more. 

Another line is 

`redonplat = ReduceLROnPlateau(monitor='val_dice_loss', mode='min', patience=10, verbose=1)`

You might know that the optimiser uses a learning rate to control how much the weights in the network are allowed to be modified. Low learning rates prevent overally drastic weight modifications. High learning rates do the opposite. Here we reduce the learning rate if there is no improvement in DICE loss after 10 epochs, and allows the learning rate to be slightly modified during training to improve overall network performance.

Although not used in thesis and publications, the line

`modalpha = AlphaScheduler(alpha, delay, reduce_alpha)`

is important if you want to modify anything in the loss function during the training procedure. For some loss functions, you can weight the amount of false positive or false negative for example, by passing in an alpha value which is passed into the `AlphaScheduler` class at the top of the script. This does have some profound impact on the overall training and can allow you to focus on training the network for a particular challenge at the start, then shift the training over many epochs to focus on a different challenge. 

The last code block of importance is
```
    history = model.fit(train_x, train_y,
        validation_data=(val_x, val_y),
        batch_size=batch,
        epochs=250,
        shuffle=True,
        callbacks=callbacks_list)
 ```
 which actually does the training process. We fit the U-net model using the training image data (`train_x`) with the training mask data (`train_y`) and also validate using validation image data (`val_x`) and validation mask data (`val_y`). Recall that these are split in the preprocessing.py script into separate folders for 75% training and 25% validation. `batch` tells you how many image samples to pass into the network at once. We don't want to pass in one image at a time because this is time consuming and images are very inconsistent between themselves. We need to network to learn some generalisability which is better when at every snapshow of training, the model is able to see multiple examples. By default this is set to 8 times the number of GPUs. An important note is that higher batch size, whilst better for training reasons, massively increases the memory demands of the system as it has to process the batch number of images at once. This value can be modified to within reason but be aware of OOM errors that might crop up. 

Everything else in the script is housekeeping and makes sure a temporary directory is created for training purposes, cleans the script and generates an output so you can see what is happening as the model learns. 

When you execute the training with

`ipython -- trainmodel.py Adam 1e-4 BCE elu 1 GM_UNet 6` 

output and progress bars will indicate the current stage of the training. If anything happens during the preprocessing of the data, the error ususally occurs within the first couple minutes. As soon as the progress bars show the actual training process, you are in the clear in terms of whether the script is executing properly. Here the model will train on the training data, then validate on the validation data. There is usually a pause during the validation so wait several minutes to make sure nothing has gone wrong. Monitor the loss values though for the training and validation steps. They should start high and over epochs they should decrease. Once you know the model is training, and training realistically well, you can leave it over night.

At the end of training there should be a new model created under the `models` directory named according to your input parameters with a `.json` and `.hdf5` file which both contain elements of the trained model. Any temporary folder created during training will be cleaned away by the `cleanup.py` script automatically. 

### Testing a trained network

With a trained model you might understandably want to check it works. The best way would be to analayse it with some metric such as image similarity if you have a manual/known mask of the result already. But for quick checking purposes you can use the model to predict on a new image. To do this you really only need the following snippets of code where you point to the `.json` and `.hdf5` file for your trained model. You then need to open the model and load the weights as below.

```
    model_path = '/home/gm515/Documents/GitHub/cell_counting_unet/models/2020_01_22_UNet_BCE_2/focal_unet_model.json'
    weights_path = '/home/gm515/Documents/GitHub/cell_counting_unet/models/2020_01_22_UNet_BCE_2/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise and compile
    print ('Loading model...')
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    print ('Done!') 
```

For prediction the only command you need is

`pred = model.predict(image)` 

where image is your loaded image. However, you might have to modify the dimensions of the image or split it up into smaller chunks. Look at `fullimageprediction.py` for an example. The key notes are that the prediction can only work on images which can be downsized the number of times that occur in the U-Net model. Therefore, the dimension of the images used for inference/prediction should be multiples of 8, or you can simply break the image in chunks of 512 x 512 pixels as we know this is already compatible as we used that image size for training.

There a lot of caveats when using Keras/Tensorflow for network learning and inference. Expect a lot of teething issues. 
