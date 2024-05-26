import time
import keras.backend as K

import numpy as np
from keras import Sequential, Input, Model
from keras.src.layers import Conv1D, Dense, AveragePooling1D, Flatten, MaxPooling1D
from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt
from numpy import shape
from tcn import TCN
from tensorflow.python.framework.ops import convert_to_tensor


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

class CNNModel:
    def __init__(self, data_extractor):
        self.data_extractor = data_extractor
        self.respiration_data, self.bcg_data = self.data_extractor.get_data()

    def run(self):

        model = None
        try:
            model = load_model('cnn/models/CNNModel3_larger_batch.h5')
            print(model.summary())
        except:
            print("Could not load the Model")

        resp_data = self.respiration_data[0]
        bcg_data = self.bcg_data[0]

        #for i in range(1, len(self.respiration_data)):
        #    resp_data = np.concatenate((resp_data, self.respiration_data[i]))
        #    bcg_data = np.concatenate((bcg_data, self.bcg_data[i]))

        resp_train, resp_test = convert_to_tensor(resp_data[0:len(resp_data) - 1]), convert_to_tensor(
            [resp_data[len(resp_data) - 1]])
        bcg_train, bcg_test = convert_to_tensor(bcg_data[0:len(bcg_data) - 1]), convert_to_tensor(
            [bcg_data[len(bcg_data) - 1]])

        if model is None:
            model = Sequential()
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(3000, 1), padding='same'))
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
            model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
            model.add(Conv1D(filters=32, kernel_size=3, padding='same'))
            model.add(Flatten())
            model.add(Dense(2000))
            model.add(Dense(3000))
            model.compile(optimizer='adam', loss='mse')
            print(model.summary())
            start = time.time()
            history = model.fit(resp_train, bcg_train, epochs=3000, batch_size=8)
            end = time.time()
            training_time = end - start
            print('The training took ' + str(training_time) + ' seconds')
            plt.plot(history.history['loss'])
            plt.show()

            model.save('cnn/models/CNNModel3_larger_batch.h5')

        test_predict = model.predict(resp_test)
        print(shape(test_predict))

        test_predict = test_predict.flatten()

        plt.plot(bcg_test[0], 'red')
        plt.plot(test_predict, 'blue')
        plt.plot(resp_test[0], 'green')
        plt.title("Sine Wave with Smooth Random Amplitude Variation")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

