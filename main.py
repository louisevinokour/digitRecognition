import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def get_data():
    # load data from mnist dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # standardize data: reshape images, so that all images are of height: 28, width: 28, & depth: 1
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # convert data to floats
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize RGB values
    x_train /= 255
    x_test /= 255

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def create_model():
    x_train, y_train, x_test, y_test = get_data()

    # initialize number of classes to 10 (1 for each digit)
    num_classes = 10

    # create model
    # include dropout: exclude 20% of nodes in next layer
    model = Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # compile model
    model.compile(keras.optimizers.Adam(),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # train model
    epochs = 10
    model.fit(
        x_train, y_train,
        batch_size=64,
        validation_data=(x_test, y_test),
        epochs=epochs
    )

    # save model
    model.save('model.h5')

if __name__=='__main__':
    if not os.path.exists('model.h5'):
        create_model()
    model = keras.models.load_model('model.h5')

    _, _, x_test, y_test = get_data()
    print(model.evaluate(x_test, y_test, False))

