# Semantic segmentation 
This is a "fork" of official TF2 tutorial <BR>
Original source URL = 'https://www.tensorflow.org/tutorials/images/segmentation' <BR>

Intended use. Create a retrainable semantic segmentation model to use on Coral TPU Dev board <BR>

The project implments Fully Convolutional Network based on pretrained Keras MobileNetV2(128, 128, 3) + ReTrainable Pix2Pix upStack.
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). Uses a pretrained model—MobileNetV2—as the encoder. For the decoder, you will use the upsample block, which is already implemented in the pix2pix example in the TensorFlow Examples.
See Original Source URL for more details.

## Step1<BR>
The program is the Collection of the scripts from the tutorial. 
It loads PETS Dataset and pretrained MobileNet V2 from internet.
Retrains UpStsack Pix2Pix layers.
Validates and shows model performance.
At the end it saves retrained model to my_model.h5

## Step2<BR>
Loading saved model and doing prediction on the same dataset.
 
## Step3<BR>  
Converting my_model.mmh5 to TFLite2 format model.tflite.2

## Step4  <BR>
Using TFLite model to perform prediction on cat1.jpg image.
<BR>
<BR>

