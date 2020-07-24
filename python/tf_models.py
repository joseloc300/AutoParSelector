import tensorflow as tf
import numpy as np


def run_tf_models(x_train, x_test, y_train, y_test):
    neural_network_tf(x_train, x_test, y_train, y_test)


def neural_network_tf(x_train, x_test, y_train, y_test):
    print("\ntensorflow")
    print(tf.__version__)

    tf.keras.backend.set_floatx('float64')

    tf_x_train = np.asarray(x_train[:])
    tf_y_train = np.asarray(y_train[:])
    tf_x_test = np.asarray(x_test[:])
    tf_y_test = np.asarray(y_test[:])

    tf_x_train = tf.keras.utils.normalize(tf_x_train, axis=1)
    tf_x_test = tf.keras.utils.normalize(tf_x_test, axis=1)

    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(6200, activation='relu', kernel_initializer='normal', input_shape=tf_x_test[0].shape),
        tf.keras.layers.Dense(62, activation='relu', kernel_initializer='normal', input_shape=tf_x_test[0].shape),
        tf.keras.layers.Dense(62, activation='relu', kernel_initializer='normal'),
        tf.keras.layers.Dense(62, activation='relu', kernel_initializer='normal'),
        tf.keras.layers.Dense(62, activation='relu', kernel_initializer='normal'),
        tf.keras.layers.Dense(62, activation='relu', kernel_initializer='normal'),
        tf.keras.layers.Dense(62, activation='relu', kernel_initializer='normal'),
        tf.keras.layers.Dense(1, activation='linear', kernel_initializer='normal')
    ])

    # optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

    # model.fit(tf_x_train, tf_y_train, epochs=500, batch_size=8, validation_data=(tf_x_test, tf_y_test))
    model.fit(tf_x_train, tf_y_train, epochs=50, batch_size=1)

    model.evaluate(tf_x_test, tf_y_test, verbose=1)
