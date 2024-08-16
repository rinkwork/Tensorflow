import tensorflow as tf
import numpy as np

a = tf.constant([[1, 2],
                 [3, 4]])

b = tf.constant([[1, 2],
                 [3, 4]])

print(f'a @ b = {tf.matmul(a,b)}')   # @ = matrix multiplication symbol / or @ = matrix multiplication operator
print(f'a @ b = {a @ b}') # @ is only for square matrix don't work for mismatched shapes

c = tf.constant([[1, 2, 3],[4, 5, 6]])

# reshape tensor
print(f'reshaped tensor = {tf.reshape(c, shape=(3,2))}')

# transpose of matrix
print(f'transpose of c = {tf.transpose(c)}')

# first transpose of one matrix and then matrix multiplication
print(f'c transpose @ b first method = {tf.matmul(tf.transpose(a), b)}')

# direct function for this
print(f'c transpose @ b second method = {tf.matmul(a=a, b=b, transpose_a=True, transpose_b=False)}')

# tensor dot product (in some cases is equal to matrix multiplication)

X = tf.constant([[1, 2],[3, 4],[5, 6]])
Y = tf.constant([[7, 8, 9],[10, 11, 12]])

print(f'tensor X dot product tensor Y = {tf.tensordot(X, Y, axes=1)}')

# type cast on tensor
d = tf.constant([1,2,3])
print(f'default type = {d.dtype}')

d_float = tf.cast(d, dtype=tf.float16)
print(f'type change into float= {d_float.dtype}')

# absolute value
e = tf.constant([-1.3, 3.3, -8.9])
print(f'absolute value of e = {tf.abs(e)}')

# finding min, max, mean, sum of tensor
Z = tf.constant(np.random.randint(low=0, high=10, size=(2,3)))
print(f'tensor Z = {Z}')

print(f'min value in Z = {tf.reduce_min(Z)}')
print(f'max value in Z = {tf.reduce_max(Z)}')
print(f'mean value in Z = {tf.reduce_mean(Z)}')
print(f'Sum of all element in Z = {tf.reduce_sum(Z)}')

# finding positional maximum and minimum

f = tf.constant([1,2,4,9,18,3,7,9,0,6])

print(f'maximum valued element position in f = {tf.argmax(f)}')
print(f'minimum valued element position in f = {tf.argmin(f)}')

# squeezing a tensor
S = tf.constant(np.random.randint(0, 100, 50), shape=(1, 1, 1, 1, 50))
print(f'S.shape, S.ndim = {S.shape},{S.ndim}')

S_squeezed = tf.squeeze(S)
print(f'S_squeezed.shape, S_squeezed.ndim = {S_squeezed.shape},{S_squeezed.ndim}')

# one-hot encoding

some_list = [0, 1, 2, 3]
print(f'one-hot encoding = {tf.one_hot(some_list, depth=4)}')

# specify name or text instead of 0 and 1
print(f'one-hot encoding with text = {tf.one_hot(some_list, depth=4, on_value="on_time", off_value="off_time")}')