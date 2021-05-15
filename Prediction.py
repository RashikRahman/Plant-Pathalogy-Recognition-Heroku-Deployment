import os
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np
from skimage import transform
import cv2
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)




flags.DEFINE_list('images', '../Data/For GDrive/Test/Healthy/8ddaa5a5caa5caa8.jpg', 'path to input image')
flags.DEFINE_list('model', './Logs/Logs/model3.hdf5', 'path to input model')


def convert(np_image,shape):
    np_image = np.array(np_image).astype('float32')/255.0
    np_image = transform.resize(np_image, (shape, shape, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def model_load():
    print('tf version', tf.__version__)
    print('keras version', tf.keras.__version__)
    print('gpu is ','available' if tf.config.list_physical_devices('GPU') else 'not available')
    model = FLAGS.model
    print(model[0])
    new_model = tf.keras.models.load_model(model[0])
    return new_model

def main(_argv):
    model = model_load()
    input_size = 80
    images = FLAGS.images 
    image = cv2.imread(images[0])
    image = convert(image,input_size)
    target_names=['Healthy','Powdery','Rust']
    print(target_names[np.argmax(model.predict(image), axis=1)[0]])


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
