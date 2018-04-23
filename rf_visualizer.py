import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.misc import imread

class Layer(collections.namedtuple("Layer", ("kernel", "rel_stride", "padding"))):
  pass

class ReceptiveField(collections.namedtuple("ReceptiveField", ("receptive_field", "abs_stride", "size", "name"))):
  pass

def build_vgg16():
  layers = collections.OrderedDict()

  layers["conv1_1"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv1_2"] = Layer((3, 3), (1, 1), "SAME")
  layers["pool1"] = Layer((2, 2), (2, 2), "SAME")
  layers["conv2_1"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv2_2"] = Layer((3, 3), (1, 1), "SAME")
  layers["pool2"] = Layer((2, 2), (2, 2), "SAME")
  layers["conv3_1"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv3_2"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv3_3"] = Layer((3, 3), (1, 1), "SAME")
  layers["pool3"] = Layer((2, 2), (2, 2), "SAME")
  layers["conv4_1"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv4_2"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv4_3"] = Layer((3, 3), (1, 1), "SAME")
  layers["pool4"] = Layer((2, 2), (2, 2), "SAME")
  layers["conv5_1"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv5_2"] = Layer((3, 3), (1, 1), "SAME")
  layers["conv5_3"] = Layer((3, 3), (1, 1), "SAME")
  layers["pool5"] = Layer((2, 2), (2, 2), "SAME")
#  layers["fc6_conv"] = Layer((7, 7), (1, 1), "SAME")
#  layers["fc7_conv"] = Layer((1, 1), (1, 1), "SAME")
#  layers["out"] = Layer((1, 1), (1, 1), "SAME")

  return layers


def receptive_field(layers, input_size, axis=0):
  # before the first layer is the input image
  # whose receptive field is just one pixel wide and abs_stride is one
  rf, abs_stride, size = 1, 1, input_size[axis]
  ranges = [[(i, i) for i in np.arange(size)]]

  receptive_fields = []

  for l in layers:
    kernel = layers[l].kernel[axis]
    rf = rf + (kernel - 1) * abs_stride

    rel_stride = layers[l].rel_stride[axis]

    if layers[l].padding == "SAME":
      pad_size = max(kernel - rel_stride, 0) if size % rel_stride == 0 else \
          max(kernel - (size % rel_stride), 0)
      size = int(np.ceil(float(size) / float(rel_stride)))
    elif layers[l].padding == "VALID":
      pad_size = 0
      size = int(np.ceil(float(size - kernel + 1) / float(rel_stride)))
    else:
      raise ValueError("Unknown padding scheme: %s" % layers[l].padding)

    abs_stride = rel_stride * abs_stride
#    print l, "r:", rf, "k:", kernel, "s:", abs_stride, "d:", size, "pad_size:", pad_size
    receptive_fields.append(ReceptiveField(rf, abs_stride, size, l))


    pad_low = pad_size // 2
    pad_high = pad_size - pad_low
    prev_size = len(ranges[-1])

    new_ranges = [
        (start if start >= 0 else 0,
        start + kernel - 1 if start + kernel < prev_size else prev_size - 1)
        for start in np.arange(
            -pad_low, prev_size + pad_high - kernel + 1, rel_stride)]
    new_ranges = [(ranges[-1][l][0], ranges[-1][h][1]) for l, h in new_ranges]    

    ranges.append(new_ranges)

  return ranges, receptive_fields


if __name__ == "__main__":
  layers = build_vgg16()
  img = imread("palm.jpg")
  input_size = img.shape[:2]
  color = [255, 0, 0]

  ranges_height, _ = receptive_field(layers, input_size, 0)
  ranges_width, _ = receptive_field(layers, input_size, 1)

  height = len(ranges_height[-1])
  width = len(ranges_width[-1])
  
  print("feature map dimension of last layer: %d, %d" % (height, width))
  
  while True:
    try:
      x = int(raw_input("x coordinate of neuron:"))  
      if x < 0 or x >= height:
        print("must be between 0 and %d" % (height - 1))
      else:
        break
    except ValueError:
      print("integer only!") 
    
  while True:
    try:
      y = int(raw_input("y coordinate of neuron:"))
      if y < 0 or y >= width:
        print("must be between 0 and %d" % (width - 1))
      else:
        break
    except ValueError:
      print("integer only!")


  lx, hx = ranges_height[-1][x]
  ly, hy = ranges_width[-1][y]

  img[lx, ly:hy+1, :] = color
  img[hx, ly:hy+1, :] = color
  img[lx:hx+1, ly, :] = color
  img[lx:hx+1, hy, :] = color

  plt.imshow(img)
  plt.show()
