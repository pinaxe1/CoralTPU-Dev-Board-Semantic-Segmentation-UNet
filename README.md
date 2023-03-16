# Semantic segmentation 
This is a derivative from official TF2 and Coral Dev Board tutorials <BR>
Original sources: 'https://www.tensorflow.org/tutorials/images/segmentation' <BR>
 'https://coral.ai/examples/'<BR>
 'https://coral.ai/docs/edgetpu/models-intro/#transfer-learning' <BR>
 'https://coral.ai/docs/edgetpu/compiler/'<BR>
 https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb<BR>
 

Intended use. Create a retrainable semantic segmentation model, train it and make it to run on Coral TPU Dev Board <BR>

The project implments Fully Convolutional Network based on pretrained Keras MobileNetV2(128, 128, 3) + ReTrainable U-Net upStack.
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). Uses a pretrained model—MobileNetV2—as the encoder. For the decoder, we'll use the upsample 2D blocks. See Original Source URL for more details. Model modifyed from original TF tutorial to make it compatible with CORAL TPU DEV BOARD. Since everything works steps 2, 3 and 4 are optional. After successfully performing steps 0,1,5,6,7,8 we can run edgetpu_compiler to create a model which will run on EDGE TPU.
To compile quantized model we use web based edgetpu-compiler. 
 'https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb' <BR>
To run ompiled model on Coral Dev Board do steps 7 and 8 
 

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
 
## Step07 Compile quantized model
Compile quantized TFLite model for EDGETPU and upload it to Coral Edge TPU Dev Board.
To compile the model use web based edgetpu-compiler. 
 'https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb' <BR>
 Upload the model to CoLab<BR>
 Compiile it <BR>
 Download compiled model <BR>
 Upload it to Coral Dev Board with cdp push model.edge.tflite <BR>

## Step08 Run the model on Coral Dev Board
To run the model we use semantic_segmentation.py program from
 'https://github.com/google-coral/pycoral/tree/master/examples'
 
 
<BR>
<BR>

