# Semantic segmentation 
This is a derivative from official TF2 tutorial and Coral Dev Board tutorials <BR>
Original sources: 'https://www.tensorflow.org/tutorials/images/segmentation' <BR>
 'https://coral.ai/examples/'<BR>

Intended use. Create a retrainable semantic segmentation model to use on Coral TPU Dev board <BR>

The project implments Fully Convolutional Network based on pretrained Keras MobileNetV2(128, 128, 3) + ReTrainable U-Net upStack.
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). Uses a pretrained model—MobileNetV2—as the encoder. For the decoder, we'll use the upsample 2D blocks.
See Original Source URL for more details. It is a modifyed a little from original TF tutorial code to make it compatible with CORAL TPU DEV BOARD.

## Step00_create_model_save_to_h5<BR>
Loading pretrained MobileNet V2 from internet.
Creates model and savves it to file. 

 ## Step01_Load_model_train_predict_and_save_to_h5<BR>
Loading saved model It loads PETS Dataset and 
Retrains UpStsack layers.
Validates and shows model performance.
At the end it saves trained model to my_model_trained.h5

## Step02_load_trained_TF_model_and_predict 
Loading trained model and runs prediction on the same dataset.
 
## Step03_convert_h5_to_tflite2<BR>  
Converting my_model_trained.h5 to TFLite2 format model.tflite.2 without quantisation. Just to check if the model compatible with TFLite. We are not going to use it anyway.

## Step04_load_tflite2_and_predict  <BR>
Using TFLite model to perform prediction on cat1.jpg image.
 
## Step05_convert_keras_h5_ to_fully_quantized_tflite <BR>
Converting my_model_trained.h5 to TFLite2 format model.quantized.tflite with full integer quantization

## Step06_load_tflite2_quantized_and_predict  <BR>
Using fully quantized TFLite model to perform prediction on cat1.jpg image.
 
 
<BR>
<BR>

