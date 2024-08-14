import tensorflow as tf

a = tf.constant([[[1,2],[2,1]],
                 [[3,4],[4,3]],
                 [[5,6],[6,7]],
                 [[8,9],[9,8]]])

print("Datatype of every element:", a.dtype)
print("Number of dimensions (rank):", a.ndim)
print("Shape of tensor:", a.shape)
print("Elements along axis 0 of tensor:", a.shape[0])
print("Elements along last axis of tensor:", a.shape[-1])
print("Total number of elements:", tf.size(a).numpy())