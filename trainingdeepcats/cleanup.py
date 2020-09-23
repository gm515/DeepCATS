"""
Model Training Data Clean-Up
Author: Gerald M

Deletes the training data directory.
"""

import os
import shutil

def clean():

    print ('Cleaning up directories...')

    for root, dirs, files in os.walk('./'):
        for dir in dirs:
            if '__pycache__' in dir:
                shutil.rmtree(os.path.join(root, dir))
            if 'WORKINGCOPY' in dir:
                shutil.rmtree(os.path.join(root, dir))

    print ('Done!')
