# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 23:45:20 2020
@author: SaiPrasad Padhy
Version : 1.0
"""

from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Import necessery libraries from Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Define path for trained model
MODEL_PATH ='vgg16_model.h5'

# Load the trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(70, 70))

    # Preprocessing the image
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    preds = np.argmax(result[0])
    if preds == 0:
        preds = "Buick"
    elif preds == 1:
        preds = "Chery"
    elif preds == 2:
        preds = "Citroen"
    elif preds == 3:
        preds = "Honda"
    elif preds == 4:
        preds = "Hyundai"
    elif preds == 5:
        preds = "Lexus"
    elif preds == 6:
        preds = "Mazda"
    elif preds == 7:
        preds = "Peugeot"
    elif preds == 8:
        preds = "Toyota"
    else:
        preds = "Volkswagen"

    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('predict.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)