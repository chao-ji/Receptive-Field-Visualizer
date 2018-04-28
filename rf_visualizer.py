import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.misc import imread

class Layer(collections.namedtuple("Layer", ("kernel", "rel_stride", "dilation", "padding"))):
  pass

class ReceptiveField(collections.namedtuple("ReceptiveField", ("receptive_field", "abs_stride", "size", "name"))):
  pass


def build_alexnet(fully_conv=False):
  layers = collections.OrderedDict()

  layers["conv1"] = Layer((11, 11), (4, 4), (1, 1), "VALID")
  layers["pool1"] = Layer((3, 3), (2, 2), (1, 1), "VALID")
  layers["conv2"] = Layer((5, 5), (1, 1), (1, 1), "SAME")
  layers["pool2"] = Layer((3, 3), (2, 2), (1, 1), "VALID")
  layers["conv3"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv4"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv5"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["pool5"] = Layer((3, 3), (2, 2), (1, 1), "VALID")
  if fully_conv == True:
    layers["fc6_conv"] = Layer((6, 6), (1, 1), (1, 1), "VALID")
    layers["fc7_conv"] = Layer((1, 1), (1, 1), (1, 1), "VALID")

  return layers

def build_zfnet():
  layers = collections.OrderedDict()

  layers["conv1"] = Layer((7, 7), (2, 2), (1, 1), "SAME")
  layers["pool1"] = Layer((3, 3), (2, 2), (1, 1), "SAME")
  layers["conv2"] = Layer((5, 5), (2, 2), (1, 1), "SAME")
  layers["pool2"] = Layer((3, 3), (2, 2), (1, 1), "SAME")
  layers["conv3"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv4"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv5"] = Layer((3, 3), (1, 1), (1, 1), "SAME")

  return layers

def build_vgg16(fully_conv=False):
  layers = collections.OrderedDict()

  layers["conv1_1"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv1_2"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["pool1"] = Layer((2, 2), (2, 2), (1, 1), "SAME")
  layers["conv2_1"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv2_2"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["pool2"] = Layer((2, 2), (2, 2), (1, 1), "SAME")
  layers["conv3_1"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv3_2"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv3_3"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["pool3"] = Layer((2, 2), (2, 2), (1, 1), "SAME")
  layers["conv4_1"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv4_2"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv4_3"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["pool4"] = Layer((2, 2), (2, 2), (1, 1), "SAME")
  layers["conv5_1"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv5_2"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["conv5_3"] = Layer((3, 3), (1, 1), (1, 1), "SAME")
  layers["pool5"] = Layer((2, 2), (2, 2), (1, 1), "SAME")
  if fully_conv == True:
    layers["fc6_conv"] = Layer((7, 7), (1, 1), (1, 1), "SAME")
    layers["fc7_conv"] = Layer((1, 1), (1, 1), (1, 1), "SAME")

  return layers


def receptive_field(layers, input_size, axis=0):
  # height: `axis` = 0
  # width: `axis` = 1

  # below the first layer is the input image
  # whose receptive field is just one pixel wide and abs_stride is one
  
  # `rf` (num of pixels): receptive field 
  # `abs_stride` (num of pixels): absolute stride of neurons in previous layer
  # `size` (num of pixels): spatial dimension of previous layer
  # `ranges`: list of `(low, high)` indicating receptive field ranges of neurons
  rf, abs_stride, size = 1, 1, input_size[axis]
  ranges = [[(i, i) for i in np.arange(size)]]

  receptive_fields = []

  for l in layers:
    kernel = (layers[l].kernel[axis] - 1) * layers[l].dilation[axis] + 1
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
    print l, "r:", rf, "k:", kernel, "s:", abs_stride, "d:", size, "pad_size:", pad_size
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
  layers = build_vgg16(True)
  img = imread("palm.jpg")
  input_size = img.shape[:2]
  color = [255, 0, 0]

  ranges_height, rf_h = receptive_field(layers, input_size, 0)

  ranges_width, _ = receptive_field(layers, input_size, 1)

  ranges_height = ranges_height[-1]
  ranges_width = ranges_width[-1]

  height = len(ranges_height)
  width = len(ranges_width)

  print("feature map dimension of last layer: %d, %d" % (height, width))
 
  xs, ys = [], []
  while True:
    try:
      xy = raw_input("coordinate of neuron:  `x(integer) y(integer)`\nOr press ENTER to quit:")
      if len(xy) == 0:
        break

      x, y = map(int, xy.split(" "))
      if x < 0 or x >= height:
        print("x must be between 0 and %d" % (height - 1))
      elif y < 0 or y >= width:
        print("y must be between 0 and %d" % (width - 1))
     
      xs.append(x)
      ys.append(y)
    except ValueError:
      print("must be two integers separated by single space!") 
    
  for x, y in zip(xs, ys):
    lx, hx = ranges_height[x]
    ly, hy = ranges_width[y]

    img[lx, ly:hy+1, :] = color
    img[hx, ly:hy+1, :] = color
    img[lx:hx+1, ly, :] = color
    img[lx:hx+1, hy, :] = color

  plt.imshow(img)
  plt.show()
