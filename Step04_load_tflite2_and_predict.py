import tensorflow as tf
import matplotlib.pyplot as plt

def normalize(input_image):
    return tf.cast(input_image, tf.float32) / 255.0

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite2')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = tf.io.read_file('cat1.jpg')
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, size=(128, 128))
image = normalize(image)

interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(image, axis=0))
interpreter.invoke()
pred_mask = interpreter.get_tensor(output_details[0]['index'])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image)
ax2.imshow(create_mask(pred_mask))
plt.show()