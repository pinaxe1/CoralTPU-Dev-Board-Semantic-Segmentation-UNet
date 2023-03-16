import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  #if label.ndim != 2:
  #  raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

# Load the TFLite model
#interpreter = tf.lite.Interpreter(model_path='my_saved_model.tflite')
interpreter = tf.lite.Interpreter(model_path='model.quantized.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input and output details
input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
#print ( input_details)
#print (output_details)

# Read and preprocess input image
image = tf.io.read_file('cat1.jpg')
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, size=(input_shape[1], input_shape[2]))
image=tf.cast(image,tf.uint8)


# Set input tensor to the interpreter
interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(image, axis=0))

# Run inference and get output tensor
interpreter.invoke()
pred_mask = interpreter.get_tensor(output_details[0]['index'])
#pred_mask = Image.fromarray(label_to_color_image(result).astype(np.uint8))

# Display input image and predicted mask
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image.numpy().astype(np.uint8))
#ax2.imshow(create_mask(pred_mask))
ax2.imshow(pred_mask[0])
plt.show()