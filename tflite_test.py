import numpy as np
import tensorflow as tf
from skimage import transform
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def convert(np_image,shape):
    np_image = np.array(np_image).astype('float32')/255.0
    np_image = transform.resize(np_image, (shape, shape, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

input_size = 80
images = './test.jpg'
image = cv2.imread(images)
image = convert(image,input_size)
target_names=['Healthy','Powdery','Rust']

interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

# print(interpreter.get_tensor(output_details[0]['index']))
print(target_names[np.argmax(interpreter.get_tensor(output_details[0]['index']), axis=1)[0]])
