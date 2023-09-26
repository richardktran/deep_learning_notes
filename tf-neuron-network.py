import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    return X / 255.0


def buildNetworkStructure():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


def main():
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()


    trainX = normalize(trainX)
    testX = normalize(testX)

    # Build and train model
    model = buildNetworkStructure()
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=7, validation_data=(testX, testY))
    model.save('model.h5')

    # Load model if needed
    # model = tf.keras.models.load_model('model.h5')

    # Predict
    testIndex = 9
    prediction = model.predict(testX[testIndex:testIndex+1, :, :])
    print(np.argmax(prediction))
    print(testY[testIndex:testIndex+1])

    # Show image
    plt.imshow(testX[testIndex])
    plt.show()


if __name__ == '__main__':
    main()
