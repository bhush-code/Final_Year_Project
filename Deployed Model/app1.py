from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np


# Keras
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from keras.models import load_model
from keras.preprocessing import image

from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import jsonify
import json

# Define a flask app
app = Flask(__name__)

def get_model():
    global model,graph
    model=keras.models.load_model('my_model.h5')


    print(" * Model Loaded!!")
    #graph = tf.get_default_graph()
    
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x, mode='keras')

    
    return x

print(" * Loading keras model.....")
#fix_layer0('diabetic_retinopathy.h5', [None, 224, 224,3], 'float32')
get_model()

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        next_url = request.form.get("next")

        if username in users and users[username][1] == password:
            session["username"] = username
            if next_url:
                return redirect(next_url)
            return redirect(url_for("index.html"))
    return render_template("login.html")    


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
        img=model_predict(file_path)
        prediction=model.predict(img).tolist()
        #response = {
         #   'prediction': {
         #       'dr': prediction[0][0],
         #       'nodr': prediction[0][1]
         #   }
        if prediction[0][0]>predication[0][1]:
         	response="DR is detected with {} % possibilites".format(predication[0][0]*100)
        else:
         	response="No DR"
        # Process your result for human
        #pred_class = preds#.argmax()#(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class)               # Convert to string
        return str(response) 
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)    # Serve the app with gevent
    http_server = WSGIServer(('', 8055), app)
    http_server.serve_forever()
