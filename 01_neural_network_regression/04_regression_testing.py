import numpy as np
import tensorflow as tf

# Set random seed (optional)
tf.random.set_seed(42)

# Recreate the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model (not necessary for prediction but keeping consistent)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Manually set the weights and biases copied from the training code
# Replace these values with the ones printed from your training code

# Example values (you need to replace them with actual values):
weights = np.array([[1.7250595]])  # This will be a 2D array
biases =  np.array([0.8324995])     # This will be a 1D array

# Set the weights and biases to the model's layer
model.layers[0].set_weights([weights, biases])

# Now the model is ready for prediction
predicted_value = model.predict([17.0])

print(f'Predicted value for input 17.0 = {predicted_value}')
