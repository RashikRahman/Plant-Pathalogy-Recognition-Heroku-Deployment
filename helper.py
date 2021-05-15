import os
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
from skimage import transform
import cv2
import numpy as np

def load_tflite_model():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def convert(np_image,shape):
    np_image = np.array(np_image).astype('float32')/255.0
    np_image = transform.resize(np_image, (shape, shape, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def main(image,interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_size = 80
    img = convert(image,input_size)

    target_names=['Healthy','Powdery','Rust']
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    return target_names[np.argmax(interpreter.get_tensor(output_details[0]['index']), axis=1)[0]]

