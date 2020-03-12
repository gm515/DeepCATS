"""
Flask deployment on local network for ML model predictions.

Run with gunicorn using the following:
    gunicorn --timeout 240 --bind 0.0.0.0:5000 wsgi:app
"""

# import the necessary packages
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from PIL import Image
from multiprocessing import cpu_count
import numpy as np
import tensorflow.keras.backend as K
import psutil

# Initialize Flask application and the Keras model
import flask
app = flask.Flask(__name__)

K.clear_session()
K.set_learning_phase(0)

GPU = False

if not GPU:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print ('Using CPU.')
else:
    print ('Using GPU.')

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

import keras.backend.tensorflow_backend as K
K.set_session(sess)

def load_model():
    # Set global variables for the model to allow inheritted model/graph
    global model
    global graph

    # model_path = '../models/2019_12_20_UNet_BCE_DICE/focal_unet_model.json'
    # weights_path = '../models/2019_12_20_UNet_BCE_DICE/focal_unet_weights.best.hdf5'

    model_path = '../models/2020_01_22_UNet_BCE_2/focal_unet_model.json'
    weights_path = '../models/2020_01_22_UNet_BCE_2/focal_unet_weights.best.hdf5'

    # Load the classifier model, initialise (and compile if needed)
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)

    model._make_predict_function()

@app.route("/predict", methods=["POST"])
def predict():
    # Initialise return dictionary
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # Read image in PIL format
            image = flask.request.files["image"].read()
            shape = flask.request.files["shape"].read()
            image = np.frombuffer(image, np.float32)
            shape = tuple(np.frombuffer(shape, np.int))

            image = image.reshape(shape)

            # Predict
            preds = model.predict(image)
            preds = np.squeeze(preds[0])

            # Add prediction to dictionary as a list (array does not work)
            data["predictions"] = preds.tolist()

            # Show the request was successful
            data["success"] = True

    print ('[Memory info] Usage: '+str(psutil.virtual_memory().percent)+'%')

    # Return the result
    return flask.jsonify(data)

print("* Loading Keras model and Flask starting server...")

load_model()

print("* Complete! Ready to use...")

if __name__ == "__main__":
    app.run(host='0.0.0.0')
