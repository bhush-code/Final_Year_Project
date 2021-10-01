from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.models import save_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
#MODEL_PATH = 'mymodel.h5'

# Load your trained model
#model = load_model(MODEL_PATH,compile=True)
#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')



def get_model():
    global model,graph
    model = load_model('diabetic_retinopathy.h5')
    print(" * Model Loaded!!")
    graph = tf.get_default_graph()
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/', methods=['GET'])
def index():
 	
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    with graph.as_default(): 
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size=(224, 224))
        prediction = model.predict(processed_image).tolist()
        response = {
            'prediction': {
                'dr': prediction[0][0],
                'nodr': prediction[0][1]
            }
        }
        return jsonify(response)


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
