# DeepCATS

Repository for **Deep**learning **C**ounting in **A**natomically **T**argeted **S**tructures.

## Installation

Get set up with Anaconda (Python 3) then install the deeplearnenv environment with

`conda env create -f environment.yml`

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

Additionally, the expected radius of a cell `-radius` and the number of cpus `-ncpu` can be modified by changing the appropriate value. Plus any other parameters listed in `deepcats.py`.
