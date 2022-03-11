# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:30:42 2021

@author: steph
"""

import uuid
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

from flask import Flask, render_template, request, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)

model = ResNet50(weights='imagenet')
photos = UploadSet(name='photos', extensions=IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
app.config["SECRET_KEY"] = os.urandom(24)
configure_uploads(app, upload_sets=photos)

@app.route('/', methods=['GET', 'POST'])
def upload():
    print(photos)
    print(request.files)
    print(request.method)
    if request.method == 'POST' and 'photo' in request.files:        
        filename = photos.save(request.files['photo'], name=uuid.uuid4().hex[:8] + '.jpg-')
        #filename = photos.save(request.files['photo'], "C:\\temp", name=uuid.uuid4().hex[:8] + '.')
        print("Filename1 " + filename)
        return redirect(url_for('show', filename=filename))
    return render_template('upload.html')

@app.route('/photo/<filename>')
def show(filename):
     print("function show for filename: " + filename)
     img_path = app.config['UPLOADED_PHOTOS_DEST'] + '/' + filename
     img = image.load_img(img_path, target_size=(224,224))
     x = image.img_to_array(img)
     x = x(np.newaxis, ...)
     x = preprocess_input(x)
     
     y_pred = model.predict(x)
     predictions = decode_predictions(y_pred, top=5)[0]
     url = photos.url(filename)
     return render_template('view_result.html', filename=filename, url = url, predictions=predictions )
     


