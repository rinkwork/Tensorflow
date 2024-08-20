import numpy as np
import tensorflow as tf

# create a variable tensor
I = tf.Variable(np.arange(0, 5))
print(f'I = {I}')

# assign final value or change value of element inside tensor
I.assign([0, 1, 2, 3, 50])
print(f'I after update = {I}')

# add value to every element
I.assign_add([2, 2, 2, 2, 2])
print(f'I after adding 2 = {I}')