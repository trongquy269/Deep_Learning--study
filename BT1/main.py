from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd

# Import data
df = pd.read_csv('./iris_original_2y.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Shuffle data and split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Assuming n is the number of features in your data
n = X_train.shape[1]

# 1. Create a Sequential model
model = Sequential()
model.add(Dense(1, use_bias=True, input_shape=(n,), activation=None))

# 2. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# 3. Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# 4. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', loss)
print('Test accuracy:', accuracy)