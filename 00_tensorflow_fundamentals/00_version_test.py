import tensorflow as tf

print(tf.__version__)

# Finding access to GPUs
print(tf.config.list_physical_devices('GPU'))

