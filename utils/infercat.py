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
      print("Input channels:", input_shape[-1])
      print("Output channels:", output_shape[-1])
      print("Weights:", weights.shape)
      print("Biases:", biases.shape)
      print("Activation:", activation)

  # ...
  def flatWeights(self):
    return self.weights.flatten()

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
      print("Input channels:", input_shape[-1])
      print("Output channels:", output_shape[-1])
      print("Stride:", stride)
      print("Kernel shape:", kernel_shape)
      print("Weights:", weights.shape)
      print("Biases:", biases.shape)
      print("Activation:", activation)

  # ...
  def flatWeights(self):
    return self.weights.flatten()

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