from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from scipy.misc import imread, imresize
import tensorflow as tf
import skimage.transform as st

from skimage.transform import resize
app = Flask(__name__)

MODEL_PATH = 'model.h5'

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)
K.set_session(session)

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
graph = tf.get_default_graph() # Change
print('Model loaded. Check http://127.0.0.1:5000/')

# def classify(image, model):
#     class_names = ['airplane','automobile','bird','cat','deer',
#                'dog','frog','horse','ship','truck']
#     preds = model.predict(image)
#     classification = np.argmax(preds)
#     final = pd.DataFrame({'name' : np.array(class_names),'probability' :preds[0]})
#     return final.sort_values(by = 'probability',ascending=False),class_names[classification]
def model_predict(img_path, model):

    try:
        with session.as_default():
            with session.graph.as_default():
                img = image.load_img(img_path, target_size=(32, 32,3))
                # Preprocessing the image
            #     x = image.img_to_array(img)
                # x = np.true_divide(x, 255)
                x = np.expand_dims(img, axis=0)

            #     x = preprocess_input(x, mode='caffe')

                preds = model.predict(np.array(x))
                return preds
    except Exception as ex:
        log.log('Seatbelt Prediction Error', ex, ex.__traceback__.tb_lineno)
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        

# #         image_url = request.form['image_url']
# #         image = io.imread(image_url)
        
#         image_small = st.resize(file_path, (32,32,3))
#         x = np.expand_dims(image_small.transpose(2, 0, 1), axis=0)
        
#         final,pred_class = classify(x, model)
#         print(pred_class)
#         print(final)
        
        #Store model prediction results to pass to the web page
#         message = "Model prediction: {}".format(pred_class)
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        
        number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        index = np.argsort(preds[0,:])
#         for x in range(len(number_to_class)):
#             if number_to_class[x] == 1:
#                 print(preds[0][i])
            
        # Process your result for human
        pred_class = preds.argmax(axis=-1)            # Simple argmax
#         pred_class = decode_predictions(preds, top=1) # ImageNet Decode
#         result = str(pred_class[0][1]) # Convert to string
        return str(number_to_class[index[9]])+str(" index : ")+str(pred_class)

        
    return None


if __name__ == '__main__':
    app.run()