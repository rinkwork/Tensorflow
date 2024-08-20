import numpy as np
import tensorflow as tf

# create tensor from numpy
J = tf.constant(np.array([3., 7., 10.]))
print(f'J = {J}')
print(f'type of J = {type(J)}')

# convert tensor J to numpy with np.array()
J_numpy_0 = np.array(J)
print(f'type of J_numpy_0 = {type(J_numpy_0)}')

# convert tensor J to numpy with .numpy()
J_numpy_1 = J.numpy()
print(f'type of J_numpy_1 = {type(J_numpy_1)}')

# Create a tensor from NumPy and from an array
numpy_J = tf.constant(np.array([3., 7., 10.])) # will be float64 (due to NumPy)
tensor_J = tf.constant([3., 7., 10.]) # will be float32 (due to being TensorFlow default)
print(f'type of numpy_J = {numpy_J.dtype}')
print(f'type of tensor_J = {tensor_J.dtype}')


