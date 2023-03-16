# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:58:17 2023

@author: P
"""

import tensorflow as tf
import numpy as np
import os

def normalize(input_image):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image
"""
def representative_dataset():
    file_names = ["cat2.jpg","dog2.jpg"]
    # Loop over the file names and yield a sample for each image
    for file_name in file_names:
        # Open the image file and resize it to 128x128 pixels
        image = tf.io.read_file(file_name)
        image = tf.image.decode_jpeg(image, channels=3)
        image = normalize(image)
        image = np.expand_dims(image, axis=0)
        # Yield a single sample
        yield [image]
"""
def representative_dataset():
    # Path to directory containing JPEG images
    image_dir = "C:\\WPy64-31090\\1-sem-seg\\dat\\valid\\images\\"
    # List all JPEG files in the directory
    file_names = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    # Loop over the file names and yield a sample for each image
    for file_name in file_names:
        # Open the image file and resize it to 128x128 pixels
        image = tf.io.read_file(image_dir+file_name)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, size=(128, 128))
        image = normalize(image)
        image = np.expand_dims(image, axis=0)
        # Yield a single sample
        yield [image]
        
# Load the Keras model
keras_model = tf.keras.models.load_model('my_model_trained.h5')

# Convert the model to TFLite format with full quantization
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

# Save the TFLite model to disk
with open('model.quantized.tflite', 'wb') as f:
    f.write(tflite_model)