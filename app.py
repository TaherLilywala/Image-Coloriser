from __future__ import division, print_function
# coding=utf-8
import numpy as np
import tensorflow as tf
import keras
import cv2
from keras import models
from keras.layers import MaxPool2D,Conv2D,UpSampling2D,Input,Dropout
from tensorflow.keras.models import Sequential
from keras.preprocessing.image import img_to_array
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
basepath = os.path.dirname(__file__)
MODEL_PATH = os.path.join(basepath, 'models/model_coloriser.h5')

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
# print('Model loaded. Start serving...')

#print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


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

        SIZE=160
        # Make prediction
        img = cv2.imread(file_path) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.resize(img, (SIZE, SIZE)) #resizing image
        img = img.astype('float32') / 255.0
        pred_inp=img_to_array(img)

        preds = tf.keras.preprocessing.image.array_to_img(np.clip(model.predict(pred_inp.reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3))
          
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'processed', secure_filename(f.filename))
        preds.save(file_path)

        return file_path
    return None


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
