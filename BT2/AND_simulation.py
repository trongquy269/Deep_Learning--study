import tensorflow as tf

@tf.function
def predict(x, w, b):
	return tf.nn.sigmoid(tf.matmul(x, w) + b)

@tf.function
def loss(y, y_hat):
	r"""Mean Squared Error"""
	return tf.reduce_mean(tf.square(y - y_hat))

# @tf.function
# def loss(y, y_hat):
# 	r"""Binary Cross Entropy"""
# 	return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_hat))

# @tf.function
# def loss(y, y_hat):
# 	r"""Binary Cross Entropy"""

# 	m = tf.shape(y)[0]
# 	a = tf.cast(y, tf.float64)
# 	b = tf.cast(y_hat, tf.float64)
# 	return (-1 / m) * tf.reduce_sum(a * tf.math.log(b) + (1.0 - a) * tf.math.log(1.0 - b))

# @tf.function
# def loss(y, y_hat):
#     """Binary Cross Entropy"""
#     m = tf.shape(y)[0]
#     sum_val = tf.TensorArray(tf.float32, size=m)

#     for i in range(m):
#         sum_val = sum_val.write(i, y[i] * tf.math.log(tf.cast(y_hat[i], tf.float32)) + 
#                                      (1.0 - y[i]) * tf.math.log(1.0 - tf.cast(y_hat[i], tf.float32)))

#     return (-1.0 / tf.cast(m, tf.float64)) * tf.reduce_sum(tf.cast(sum_val.stack(), tf.float64))

x = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = tf.constant([[0.0], [0.0], [0.0], [1.0]])
w = tf.Variable(tf.random.normal([2, 1]))
b = tf.Variable(tf.random.normal([1]))

alpha = 0.1
epochs = 500
for i in range(epochs):
	with tf.GradientTape() as tape:
		current_loss = loss(y, predict(x, w, b))

	print("it", i, ":", current_loss)
	dw, db = tape.gradient(current_loss, [w, b])
	w.assign_sub(alpha * dw)
	b.assign_sub(alpha * db)

y_hat = predict(x, w, b)
# Standardize the output to 0 or 1
y_hat = tf.where(y_hat < 0.5, 0.0, 1.0)
print("y_hat = ", y_hat)