# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
# Creates a classic fully connected, multilayer perceptron style machine 
# learning model for MNIST digit database
# -----------------------------------------------------------------------------
import os

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

# Convert 2D image into 1D array. We could have done achived the same 
# with a 'Flatten' layer. Doing this manually beforehand simplifies 
# the model structure a bit.
x_train = tf.reshape(x_train, shape=[-1, 784])
x_test = tf.reshape(x_test, shape=[-1, 784])

# Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ...
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, input_dim=784, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
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
  epochs=16,
  validation_split=0.1
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
score = model.evaluate(x_test, y_test, verbose=2)

# ...
print("")
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# ...
print("")
print("Model saved for later use without re-training")
