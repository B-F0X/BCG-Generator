import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.saving.saving_api import load_model

from matplotlib import pyplot as plt
from numpy import shape
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework.config import list_physical_devices
from tensorflow.python.framework.ops import convert_to_tensor

from data_extractor.DataExtractor import DataExtractor
from variables import variables


class LSTMModel:
    def __init__(self, data_extractor):
        self.data_extractor = data_extractor
        self.respiration_data, self.bcg_data = self.data_extractor.get_data()


    def run(self):

        print(f"TensorFlow has access to the following devices:\n{list_physical_devices()}")
        model = None
        try:
            model = load_model('lstm/models/lstmModel9_less_data.h5')
            print(model.summary())
        except:
            print("Could not load the Model")

        resp_data = self.respiration_data[7]
        bcg_data = self.bcg_data[7]
        #for i in range(1, len(self.respiration_data)):
        #    resp_data = np.concatenate((resp_data, self.respiration_data[i]))
        #    bcg_data = np.concatenate((bcg_data, self.bcg_data[i]))


        resp_train, resp_test = convert_to_tensor(resp_data[0:len(resp_data) - 1]), convert_to_tensor([resp_data[len(resp_data) - 1]])
        bcg_train, bcg_test = convert_to_tensor(bcg_data[0:len(bcg_data) - 1]), convert_to_tensor([bcg_data[len(bcg_data) - 1]])

        #resp_train, resp_test = convert_to_tensor(resp_data[0:20]), convert_to_tensor([resp_data[1]])
        #bcg_train, bcg_test = convert_to_tensor(bcg_data[0:20]), convert_to_tensor([bcg_data[1]])

        print(shape(resp_train))
        print(shape(bcg_train))
        print(shape(resp_test))

        if model is None:
            model = Sequential()
            model.add(LSTM(140, input_shape=(30 * 100, 1), return_sequences=True))
            model.add(LSTM(140, return_sequences=True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.build(shape(resp_train))
            print(model.summary())
            model.fit(resp_train, bcg_train, epochs=2500, batch_size=2)

            model.save('lstm/models/lstmModel9_less_data.h5')

        test_predict = model.predict(resp_test)
        print(shape(test_predict))

        test_predict = test_predict.flatten()

        #plt.plot(bcg_test[0], 'red')
        plt.plot(test_predict, 'blue')
        plt.plot(resp_test[0], 'green')
        plt.title("Sine Wave with Smooth Random Amplitude Variation")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def show_training_data(self):
        for i in range(len(self.respiration_data[0])):
            plt.plot(self.bcg_data[0][i].flatten(), 'blue')
            plt.plot(self.respiration_data[0][i].flatten(), 'red')
            plt.title("Respiration and BCG Data")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()
