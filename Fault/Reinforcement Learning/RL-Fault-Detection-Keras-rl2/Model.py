import tensorflow as tf
from keras.layers import Activation


def create_model(structure, LS):
    model = tf.keras.Sequential()
    
    # Structure 1
    if structure == 1:
        model.add(tf.keras.layers.InputLayer(input_shape=(1, 1, 784,)))

        model.add(tf.keras.layers.Reshape((28, 28, 1)))

        model.add(tf.keras.layers.Conv2D(20, 4, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Conv2D(40, 4, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(LS))#, activation='softmax'))
        model.add(Activation('linear'))

    # Structure 2
    elif structure == 2:
        model.add(tf.keras.layers.InputLayer(input_shape=(1, 1, 784,)))

        model.add(tf.keras.layers.Reshape((28, 28, 1)))

        model.add(tf.keras.layers.Conv2D(25, 3, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Conv2D(50, 3, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(LS))#, activation='softmax'))
        model.add(Activation('linear'))

    # Structure 3
    elif structure == 3:
        model.add(tf.keras.layers.InputLayer(input_shape=(1, 1, 784,)))

        model.add(tf.keras.layers.Reshape((28, 28, 1)))

        model.add(tf.keras.layers.Conv2D(30, 5, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Conv2D(60, 5, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(LS))#, activation='softmax'))
        model.add(Activation('linear'))

    # Structure 4
    elif structure == 4:
        model.add(tf.keras.layers.InputLayer(input_shape=(1, 1, 784,)))

        model.add(tf.keras.layers.Reshape((28, 28, 1)))

        model.add(tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(LS))#, activation='softmax'))
        model.add(Activation('linear'))

    return model
