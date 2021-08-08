# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
import os
import numpy as np

# ...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# ...
print("")
print("tensorflow version:", tf.version.VERSION)

# The data, split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ...
print("")
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ...
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28)),
  tf.keras.layers.GRU(128, reset_after=False, activation='relu', recurrent_activation='sigmoid', use_bias=True),
  tf.keras.layers.Dense(10, activation='softmax', use_bias=True)
])

# ...
model.compile(
  loss="categorical_crossentropy", 
  optimizer="adam", 
  metrics=["accuracy"]
)

# Train the model
print("")
model.fit(
  x_train, y_train, 
  batch_size=128, 
  epochs=2
)

# ...
model.save('mnist_model')
tf.keras.utils.plot_model(
  model, to_file='../../../docs/mnist_model.png'
)

# ...
print("")
model.summary()

# Evaluate the trained model
print("")
print("Evaluation ...")
score = model.evaluate(x_test, y_test)

# ...
print("")
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# ...
print("")
print("Model saved for later use without re-training")
