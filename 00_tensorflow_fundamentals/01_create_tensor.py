import tensorflow as tf

# create scalar
scalar = tf.constant(7)
print(scalar)

# check the number of dimensions of tensor
print(f'dimension of scalar = {scalar.ndim}')

# create vector
vector = tf.constant([10,10])

print(f'dimension of vector = {vector.ndim}')

# create matrix
matrix = tf.constant([[10, 9],
                      [4, 3]])

print(f'dimension of matrix = {matrix.ndim}')

#by default Tensorflow creates tensors with either int32 or float32 datatype

m = tf.constant([[[1,2,3,1],[4,5,6,1]],
                [[7,8,9,1],[10,11,12,1]],
                [[13,14,15,1],[16,17,18,1]]])

# shape = (3, 2, 4) , 3 = total main elements, 2 = totel elements inside main elements, 4 = total elements inside sub elements
print(m)

# constant and variable tensor
changeable_tensor =  tf.Variable([1,2])
unchangeable_tensor = tf.constant([11,22])

# change value of variable tensor
changeable_tensor[0].assign(4)

# random tensor
random =  tf.random.Generator.from_seed(10).normal(shape=(2,3))
print(random)

# shuffle a tensor
s = tf.constant([[1, 2],[3, 4]])
print(tf.random.shuffle(s))
print(tf.random.shuffle(s, seed=2))

# ones and zeros
print(f'ones = {tf.ones(shape=(3, 2))}')
print(f'zeros = {tf.zeros(shape=(4, 2))}')
