import tensorflow as tf
import pandas as pd

# Load the data
data = pd.read_csv("iris.csv", delimiter=",")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Take only the first 2 classes to make it a binary classification problem
X = X[(y == "Iris-setosa") | (y == "Iris-versicolor")]
y = y[(y == "Iris-setosa") | (y == "Iris-versicolor")]

# Convert the labels to 0 and 1
y = y.eq('Iris-setosa').mul(1)

# These functions Tensorflow graph
@tf.function
def predict(x, w, b):
	return tf.nn.sigmoid(tf.matmul(x, w) + b)

@tf.function
def loss(y, y_hat):
	r"""Binary Cross Entropy"""
	return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_hat))

# Initialize the weights and bias
w = tf.Variable(tf.random.normal((X.shape[1], 1), dtype=tf.float64))
b = tf.Variable(tf.random.normal([1], dtype=tf.float64))

# Hyperparameters
alpha = 0.1
epochs = 500

# Training loop
for epoch in range(epochs):
	with tf.GradientTape() as tape:
		y_hat = predict(X, w, b)
		current_loss = loss(y, y_hat)

	print("it", epoch, ":", current_loss)
	dw, db = tape.gradient(current_loss, [w, b])

	# Update the weights and bias
	w.assign_sub(alpha * dw)
	b.assign_sub(alpha * db)

# Predictions
y_hat = predict(X, w, b)

# Threshold predictions and calculate accuracy
threshold = 0.5
y_hat = tf.where(y_hat < threshold, 0, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_hat), tf.float64))
print("y_hat = ", y_hat)
print("accuracy = ", accuracy.numpy())