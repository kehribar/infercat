# -----------------------------------------------------------------------------
# 
# 
# -----------------------------------------------------------------------------
import os
import sys
import infercat
import numpy as np
from utils import *

# ...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# -----------------------------------------------------------------------------
def generateCustomLayers(model):

  # ...
  layer_list = []

  # ...
  for ind, layer in enumerate(model.layers):

    # ...
    print("")
    print("LAYER", ind)
    print("-------")
    print("Tensorflow name:", layer.name)

    # ...
    if(isinstance(layer, tf.keras.layers.Dense)):
     
      # ...
      dense_layer = infercat.Dense(
        layer.input_shape[1:], 
        layer.output_shape[1:], 
        layer.get_weights()[0], 
        layer.get_weights()[1], 
        tf_getLayerActivationName(layer), 
        "%s_%s" % (modelname, layer.name)
      )

      # ...
      layer_list.append(dense_layer)

    # ...
    elif(isinstance(layer, tf.keras.layers.Conv2D)):
     
      # ...
      conv2d_layer = infercat.Conv2D(
        layer.input_shape[1:], 
        layer.output_shape[1:], 
        layer.strides, 
        layer.kernel_size,
        layer.get_weights()[0], 
        layer.get_weights()[1], 
        tf_getLayerActivationName(layer), 
        "%s_%s" % (modelname, layer.name)
      )

      # ...
      layer_list.append(conv2d_layer)

    # ...
    elif(isinstance(layer, tf.keras.layers.MaxPooling2D)):

      # ...
      maxPooling2D_layer = infercat.MaxPooling2D(
        layer.input_shape[1:], 
        layer.output_shape[1:], 
        layer.strides, 
        layer.pool_size, 
        "%s_%s" % (modelname, layer.name)
      )

      # ...
      layer_list.append(maxPooling2D_layer)

    # ...
    elif(isinstance(layer, tf.keras.layers.Flatten)):
      pass

    # ...
    else:
      print("Export supported: NO")

  # ...
  return layer_list

# -----------------------------------------------------------------------------
if __name__ == '__main__':

  # Load pre-trained model
  print("Tensorflow version:", tf.version.VERSION)
  model = tf.keras.models.load_model(sys.argv[1])
  model.summary()

  # Export file
  fp = open(sys.argv[2], "w+")
  modelname = os.path.split(sys.argv[2])[-1]
  modelname = modelname.rsplit(".")[0]

  # ...
  layer_list = generateCustomLayers(model)
  exportCustomLayersToFile(fp, modelname, layer_list)
  fp.close()

  # ...
  print("")
  print("Script exits.")
