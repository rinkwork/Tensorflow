import tensorflow
import tensorflow as tf

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
