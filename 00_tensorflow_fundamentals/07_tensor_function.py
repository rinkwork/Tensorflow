import tensorflow as tf
import numpy as np

# Create a simple function
def function(x, y):
  return x + y

# Create the same function and decorate it with tf.function (more speed in terms of execution)
@tf.function
def tf_function(x, y):
  return x + y

x = tf.constant(np.arange(0, 10))
y = tf.constant(np.arange(10, 20))
print(f'x + y using normal function= {function(x, y)}')
print(f'x + y using tf.function with decorators = {tf_function(x, y)}')



