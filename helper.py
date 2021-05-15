import os
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
from skimage import transform
import cv2
import numpy as np

def load_model():
    model = tf.keras.models.load_model('./Logs/Logs/model3.hdf5')
    return model

def convert(np_image,shape):
    np_image = np.array(np_image).astype('float32')/255.0
    np_image = transform.resize(np_image, (shape, shape, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def main(image,model):
    input_size = 80
    img = convert(image,input_size)
    target_names=['Healthy','Powdery','Rust']
    return target_names[np.argmax(model.predict(img), axis=1)[0]]

