import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# create features
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels (using tensors)
Y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# plot
plt.scatter(X, Y)
plt.show()

