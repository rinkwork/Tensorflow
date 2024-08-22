import tensorflow as tf

# Set random seed
tf.random.set_seed(42)

# Create a model using the Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Create features
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

# Extract the weights and biases
weights, biases = model.layers[0].get_weights()

# Print the weights and biases for copying
print("Weights:", weights)
print("Biases:", biases)

print(f'type of Weights = {type(weights)}')
print(f'type of Biases = {type(biases)}')