import numpy as np
import pandas as pd

data = pd.read_csv('digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Create dev set
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.0

# Create Training set
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape


# Initialize parameters
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# Activation functions ReLU and Softmax
def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
    return A


# Forward propagation
def f_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# ReLU derivative
def relu_d(Z):
    return Z > 0


# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


# Backpropagation
def b_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    dz2 = A2 - one_hot_Y
    dw2 = np.dot(dz2, A1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.dot(W2.T, dz2) * relu_d(Z1)
    dw1 = np.dot(dz1, X.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    return dw1, db1, dw2, db2


# Update parameters
def update_param(W1, b1, W2, b2, dw1, db1, dw2, db2, alpha):
    W1 -= alpha * dw1
    b1 -= alpha * db1
    W2 -= alpha * dw2
    b2 -= alpha * db2
    return W1, b1, W2, b2


# Predictions and accuracy
def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


# Gradient descent
def g_descent(X, Y, alpha, it):
    W1, b1, W2, b2 = init_params()
    for i in range(it):
        Z1, A1, Z2, A2 = f_prop(W1, b1, W2, b2, X)
        dw1, db1, dw2, db2 = b_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_param(W1, b1, W2, b2, dw1, db1, dw2, db2, alpha)

        if i % 10 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration {i}, Accuracy: {accuracy}")

    return W1, b1, W2, b2


# Run gradient descent
W1, b1, W2, b2 = g_descent(X_train, Y_train, 0.10, 500)


# Making Actual Predictions

def make_pred(X, W1, b1, W2, b2):
    _, _, _, A2 = f_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


import matplotlib.pyplot as plt

def test_prediction(index, W1, b1, W2, b2):
    # Reshape and normalize a single image for prediction
    current_image = X_train[:, index].reshape(-1, 1)  # Reshape to (784, 1)
    prediction = make_pred(current_image, W1, b1, W2, b2)  # Get prediction
    label = Y_train[index]  # True label

    # Display prediction and true label
    print("Prediction:", prediction)
    print("Label:", label)

    # Reshape to 28x28 for display
    current_image_display = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image_display, interpolation='nearest')
    plt.show()

test_prediction(4, W1, b1, W2, b2)


