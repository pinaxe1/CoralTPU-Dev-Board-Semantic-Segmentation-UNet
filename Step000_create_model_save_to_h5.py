# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:23:11 2023

@author: P

Create UNet model withou pix2pix lib 
and Conv2DTranspose layer which are not compatible with TPU
"""

import tensorflow as tf

# Define the input shape
inputs = tf.keras.layers.Input(shape=[128, 128, 3])

# Define the encoder part of the model using MobileNetV2
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
encoder_outputs = [
    base_model.get_layer('block_1_expand_relu').output,   # 64x64
    base_model.get_layer('block_3_expand_relu').output,   # 32x32
    base_model.get_layer('block_6_expand_relu').output,   # 16x16
    base_model.get_layer('block_13_expand_relu').output,  # 8x8
    base_model.get_layer('block_16_project').output,      # 4x4
]
encoder = tf.keras.Model(inputs=base_model.input, outputs=encoder_outputs)
encoder.trainable = False

# Define the decoder part of the model using upsampling layers and skip connections
decoder_inputs = encoder(inputs)
x = decoder_inputs[-1]
for i in range(len(decoder_inputs)-2, -1, -1):
    x = tf.keras.layers.UpSampling2D(2)(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, decoder_inputs[i]])
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D(2)(x)
x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(3, 1, padding='same')(x)


# Define the model
model = tf.keras.Model(inputs=inputs, outputs=x)
# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

# Save the model to a file
model.save('my_model.h5')