# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 22:15:38 2023

@author: P
"""

import tensorflow as tf

# Load the Keras model in HDF5 format
keras_model = tf.keras.models.load_model('my_model.h5')

# Convert the Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite2', 'wb') as f:
    f.write(tflite_model)