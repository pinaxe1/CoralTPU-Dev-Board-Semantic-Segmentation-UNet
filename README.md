# Semantic segmentation 
This is a "fork" of official TF2 tutorial <BR>
Original source URL = 'https://www.tensorflow.org/tutorials/images/segmentation' <BR>

Intended use. Create a retrainable semantic segmentation model to use on Coral TPU Dev board <BR>

The project implments Fully Convolutional Network based on pretrained Keras MobileNetV2(128, 128, 3) + ReTrainable Pix2Pix upStack.
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). Uses a pretrained model—MobileNetV2—as the encoder. For the decoder, you will use the upsample block, which is already implemented in the pix2pix example in the TensorFlow Examples.
See Original Source URL for more details. It is a modifyed a little from original TF tutorial code to make it compatible with CORAL TPU DEV BOARD.

## Step01<BR>
The program is the Collection of the scripts from the tutorial. 
It loads PETS Dataset and pretrained MobileNet V2 from internet.
Retrains UpStsack Pix2Pix layers.
Validates and shows model performance.
At the end it saves retrained model to my_model.h5

## Step02<BR>
Loading saved model and doing prediction on the same dataset.
 
## Step03_convert_h5_to_tflite2<BR>  
Converting my_model_trained.h5 to TFLite2 format model.tflite.2 without quantisation. Just to check if the model compatible with TFLite. We are not going to use it anyway.
 
## Step03_convert_keras_h5_ to_fully_quantized_tflite <BR>
Converting my_model_trained.h5 to TFLite2 format #model.quantized.tflite with full integer quantization

## Step04_load_tflite2_and_predict  <BR>
Using TFLite model to perform prediction on cat1.jpg image.## 

## Step04_load_tflite2_quantized_and_predict  <BR>
Using fully quantized TFLite model to perform prediction on cat1.jpg image.## 
 
 
<BR>
<BR>

