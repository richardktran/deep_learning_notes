import numpy as np
import tensorflow as tf

LEARNING_RATE = 0.1
EPOCHS = 4000


def load_data():
    """
    Get mnist dataset from tensorflow, 6000 images for training, 1000 images for validation
    X_train: (n, 28, 28)
    Y_train: (n,)
    """
    return tf.keras.datasets.mnist.load_data()


def init_params(d0, d1, d2, d3):
    W1 = np.random.randn(d0, d1) * 0.01
    b1 = np.zeros((1, d1))
    W2 = np.random.randn(d1, d2) * 0.01
    b2 = np.zeros((1, d2))
    W3 = np.random.randn(d2, d3) * 0.01
    b3 = np.zeros((1, d3))

    return W1, b1, W2, b2, W3, b3


def preprocess_image(images):
    # Flatten the images
    no_images = images.shape[0]
    images = np.reshape(images, (no_images, -1))
    # Normalize the images
    normalized_images = images / 255.0

    # Shape of images: (n, 784)
    return normalized_images


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)


def softmax(Z):
    # Z: (n, d3) = (n, 10)
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = e_Z / e_Z.sum(axis=1, keepdims=True)
    return A


def cross_entropy_loss(y_hat, y):
    # y_hat: (n, 10)
    # y: (n, 10)
    n = y.shape[0]
    return (-1 / n) * np.sum(y * np.log(y_hat))


def predict(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(X, W1) + b1  # (n, d0) x (d0, d1) = (n, d1) = (6000, 1024)
    A1 = relu(Z1)  # (n, d1) = (6000, 1024)
    Z2 = np.dot(A1, W2) + b2  # (n, d1) x (d1, d2) = (n, d2) = (n, 256)
    A2 = relu(Z2)  # (n, d2) = (n, 256)
    Z3 = np.dot(A2, W3) + b3  # (n, d2) x (d2, d3) = (n, d3) = (n, 10)
    A3 = softmax(Z3)  # (n, d3) = (n, 10)

    return Z1, A1, Z2, A2, Z3, A3, np.argmax(A3, axis=1)


def train_model(X_train, Y_train, X_val, Y_val, W1, b1, W2, b2, W3, b3, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    consts = []
    n = X_train.shape[0]
    for epoch in range(epochs):
        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2,
            'W3': W3,
            'b3': b3
        }
        # Forward propagation (784 mean 28x28)
        Z1, A1, Z2, A2, Z3, A3, _ = predict(X_train, parameters)  # (n, 1024), (n, 256), (n, 10)

        # Convert labels to one-hot vectors
        y_train_one_hot = np.eye(10)[Y_train]  # (n, 10)

        # Compute loss
        const = cross_entropy_loss(A3, y_train_one_hot)
        accuracy = np.mean(np.argmax(A3, axis=1) == Y_train)

        _, _, _, _, _, _, val_predict = predict(X_val, parameters)
        accuracy_val = np.mean(val_predict == Y_val)
        consts.append(const)

        # Backward propagation
        E3 = (A3 - y_train_one_hot) / n  # (n, 10) - (n, 10) = (n, 10) = (n, d3)
        dW3 = np.dot(A2.T, E3)  # (d2, n) x (n, d3) = (d2, d3) = (256, 10)
        db3 = np.sum(E3, axis=0, keepdims=True)  # (1, 10)
        E2 = np.dot(E3, W3.T) * relu_derivative(Z2)  # (n, d3) x (d3, d2) * (n, d2) = (n, d2) = (n, 256)
        dW2 = np.dot(A1.T, E2)  # (d1, n) x (n, d2) = (d1, d2) = (1024, 256)
        db2 = np.sum(E2, axis=0, keepdims=True)  # (1, 256)
        E1 = np.dot(E2, W2.T) * relu_derivative(Z1)  # (n, d2) x (d2, d1) * (n, d1) = (n, d1) = (n, 1024)
        dW1 = np.dot(X_train.T, E1)  # (d0, n) x (n, d1) = (d0, d1) = (784, 1024)
        db1 = np.sum(E1, axis=0, keepdims=True)  # (1, 1024)

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        print(f'Epoch {epoch + 1}/{epochs}, loss: {const}, Accuracy: {accuracy}, Accuracy val: {accuracy_val}')

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }

    return parameters, consts


def main():
    (X_train, Y_train), (X_val, Y_val) = load_data()

    # Flatten and normalize the images
    X_train = preprocess_image(X_train)  # (n, 784)
    X_val = preprocess_image(X_val)  # (n, 784)
    n = X_train.shape[0]
    no_categories = 10
    d0 = X_train.shape[1]  # 784
    d1 = 1024
    d2 = 256
    W1, b1, W2, b2, W3, b3 = init_params(d0, d1, d2, no_categories)

    parameters, consts = train_model(X_train, Y_train, X_val, Y_val, W1, b1, W2, b2, W3, b3, EPOCHS, LEARNING_RATE)


if __name__ == '__main__':
    main()
