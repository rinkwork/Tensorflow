import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import plot_model

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend()


X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4) # y = x + 10

# Split data into training/test set
# One of the other most common and important steps in a machine learning project is creating a training and test set (and when required, a validation set).

# Training set - 70% to 80% of total data
# Validation set - 10% to 15% of total data
# test set - 10% to 15% of total data

# Split data into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

#print(f'length of X_train = {len(X_train)}, length of X_test = {len(X_test)}')

plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(X_train, y_train, c='b', label='Training data')
# Plot test data in green
plt.scatter(X_test, y_test, c='g', label='Testing data')
# Show the legend
plt.legend()


# Set random seed
tf.random.set_seed(42)

# Create a model (same as above)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1]) # define the input_shape to our model
])


# Compile model (same as above)
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])


#plt.show()

# Total params - total number of parameters in the model.
# Trainable parameters - these are the parameters (patterns) the model can update as it trains.
# Non-trainable parameters - these parameters aren't updated during training
# (this is typical when you bring in the already learned patterns from other models during transfer learning).

# Fit the model to the training data
model.fit(X_train, y_train, epochs=100, verbose=0) # verbose controls how much gets output

print(model.summary())

#plot_model(model, show_shapes=True)

# Make predictions
y_preds = model.predict(X_test)

# View the predictions
print(f'y_preds = {y_preds}')

plot_predictions(train_data=X_train,
                 train_labels=y_train,
                 test_data=X_test,
                 test_labels=y_test,
                 predictions=y_preds)




