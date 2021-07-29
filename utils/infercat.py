# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import numpy as np
from utils import *

# -----------------------------------------------------------------------------
DEBUG = True

# -----------------------------------------------------------------------------
class Dense:

  # ...
  def __init__(
    self, input_shape, output_shape,
    weights, biases, activation, name
  ):

    # ...
    self.name = name
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.weights = weights
    self.biases = biases
    self.activation = activation

    # ...
    if DEBUG:
      print("Name:", name)
      print("Input shape:", input_shape)
      print("Output shape:", output_shape)
      print("Weights:", weights.shape)
      print("Biases:", biases.shape)
      print("Activation:", activation)

  # ...
  def flatWeights(self):
    data = self.weights
    data = np.reshape(data, data.shape[0] * data.shape[1])
    return data

# -----------------------------------------------------------------------------
class Conv2D:

  # ...
  def __init__(
    self, input_shape, output_shape,
    stride, kernel_shape, weights, biases,
    activation, name
  ):

    # ...
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.stride = stride
    self.kernel_shape = kernel_shape
    self.weights = weights
    self.biases = biases
    self.activation = activation
    self.name = name
    
    # ...
    if DEBUG:
      print("Name:", name)
      print("Input shape:", input_shape)
      print("Output shape:", output_shape)
      print("Stride:", stride)
      print("Kernel shape:", kernel_shape)
      print("Weights:", weights.shape)
      print("Biases:", biases.shape)
      print("Activation:", activation)

  # ...
  def flatWeights(self):
    
    # ...
    w = self.weights.shape[0] # width
    h = self.weights.shape[1] # height
    i = self.weights.shape[2] # input size
    o = self.weights.shape[3] # output size

    # ...
    ind = 0
    data = np.zeros(w * h * i * o)

    # ...
    for oo in range(0, o):
      for ii in range(0, i):
        for ww in range(0, w):
          for hh in range(0, h):
            data[ind] = self.weights[ww, hh, ii, oo]
            ind += 1

    # ...
    return data

# -----------------------------------------------------------------------------
class MaxPooling2D:

  # ...
  def __init__(
    self, input_shape, output_shape, 
    stride, pool_shape, name
  ):

    # ...
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.stride = stride
    self.pool_shape = pool_shape
    self.name = name

    # ...
    if DEBUG:
      print("Name:", name)
      print("Input shape:", input_shape)
      print("Output shape:", output_shape)
      print("Stride:", stride)
      print("Pool shape:", pool_shape)