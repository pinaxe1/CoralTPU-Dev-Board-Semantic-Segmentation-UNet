# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Derived from 
import cv2
import numpy as np
from PIL import Image
import time

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter


def create_pascal_label_colormap():

  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3
    colormap[0]=[0,0,255]
  return colormap


def label_to_color_image(label):

  colormap = create_pascal_label_colormap()
  return colormap[label]

def main():
  print("",time.time())
  interpreter = make_interpreter('./mod_edgetpu.tflite', device=':0')
  interpreter.allocate_tensors()
  width, height = common.input_size(interpreter)
  start=time.time()
  print("Start ",time.time()-start)
  start=time.time()
  names=("cat1.jpg","cat2.jpg","dog1.jpg","dog2.jpg")
  im=[]
  for fname in names: 
      img=Image.open(fname)
      resized_img = img.resize((width, height), Image.ANTIALIAS)
      im.append(resized_img)
  print("Load ",time.time()-start)
  start=time.time()
  for resized_img in im :
      common.set_input(interpreter, resized_img)
      interpreter.invoke()
      result = segment.get_output(interpreter)
      print("Result ",time.time()-start)
      start=time.time()
      if len(result.shape) == 3:
        result = np.argmax(result, axis=-1)

      new_width, new_height = resized_img.size
      print("Resize ",time.time()-start)
      start=time.time()
      result = result[:new_height, :new_width]
      mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))
    
      # Concat resized input image and processed segmentation results.
      output_img = Image.new('RGB', (2 * new_width, new_height))
      output_img.paste(resized_img, (0, 0))
      output_img.paste(mask_img, (width, 0))

      #convert to CV for visualisation
      numpy_im=np.array(output_img)
      cv2.imshow("Image",numpy_im)
      print("Show ",time.time()-start)
      start=time.time()
      cv2.waitKey(0)
      print("Tupim ",time.time()-start)
      start=time.time()
  cv2.destroyAllWindows()
if __name__ == '__main__':
  main()
