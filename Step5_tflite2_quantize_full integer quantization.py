# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:41:10 2023

@author: TENSORFLOW
@Source: https://www.tensorflow.org/tutorials/images/segmentation
"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image



import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt




#model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
#model.compile(optimizer='adam',
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])
#
#model.save('my_model.h5')  # Save the model to a file

def normalize(input_image):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image

def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image = normalize(input_image)

  return input_image

def representative_dataset():
    # Path to directory containing JPEG images
    image_dir = "C:\\WPy64-31090\\1-sem-seg\\dat\\test\\images\\"
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
            
model = tf.keras.models.load_model('my_model.h5')
# Convert the model to a SavedModelflat buffer with the ".pb" extension
tf.saved_model.save(model, 'my_saved_model')

# Convert the model to TensorFlow Lite format with full integer quantization
converter = tf.lite.TFLiteConverter.from_saved_model('my_saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset  # A dataset that covers the input range
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the converted model to a file
with open("saved_model.tflite", "wb") as f:
    f.write(tflite_model)            