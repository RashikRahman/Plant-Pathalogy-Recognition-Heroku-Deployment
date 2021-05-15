import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('./Notebook/Logs/saved_model/my_model') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('./Notebook/Logs/Logs/model.tflite', 'wb') as f:
  f.write(tflite_model)