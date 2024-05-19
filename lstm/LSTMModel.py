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

from variables import variables


class LSTMModel:
    def run(self):
        respiration = pd.read_csv(variables.path_to_respiration_data + 'resp1.csv')
        bcg = pd.read_csv(variables.path_to_bottom_right_bcg + 'bcg_bottom_right1.csv')

        print(f"TensorFlow has access to the following devices:\n{list_physical_devices()}")
        model = None
        try:
            model = load_model('lstm/models/lstmModel7_GPU.h5')
            print(model.summary())
        except:
            print("Could not load the Model")

        print(len(respiration['data']))
        print(len(bcg['data']))

        # respiration['C'] = range(len(respiration))

        respiration_array = respiration['data'].to_numpy()
        bcg_array = bcg['data'].to_numpy()
        respiration_array = respiration_array[::10]
        bcg_array = bcg_array[::10]
        respiration_array = np.array([[num] for num in respiration_array])
        bcg_array = np.array([[num] for num in bcg_array])

        print(len(respiration_array))
        print(len(bcg_array))

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(-1, 1))
        respiration_array = scaler.fit_transform(respiration_array)
        bcg_array = scaler.fit_transform(bcg_array)

        values_per_minute = int(len(respiration_array) / 10)
        resp_data = []
        bcg_data = []
        for i in range(0, 4 * values_per_minute, 200):
            resp_data.append(respiration_array[i:i + values_per_minute])
            bcg_data.append(bcg_array[i:i + values_per_minute])

        #resp_train, resp_test = convert_to_tensor(resp_data[0:len(resp_data) - 1]), convert_to_tensor([resp_data[len(resp_data) - 1]])
        #bcg_train, bcg_test = convert_to_tensor(bcg_data[0:len(bcg_data) - 1]), convert_to_tensor([bcg_data[len(bcg_data) - 1]])

        resp_train, resp_test = convert_to_tensor(resp_data[0:20]), convert_to_tensor([resp_data[1]])
        bcg_train, bcg_test = convert_to_tensor(bcg_data[0:20]), convert_to_tensor([bcg_data[1]])

        print(shape(resp_train))
        print(shape(bcg_train))
        print(shape(resp_test))

        if model is None:
            model = Sequential()
            model.add(LSTM(140, input_shape=(values_per_minute, 1), return_sequences=True))
            model.add(LSTM(140, return_sequences=True))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.build(shape(resp_train))
            print(model.summary())
            model.fit(resp_train, bcg_train, epochs=1000, batch_size=2)

            model.save('lstm/models/lstmModel7_GPU.h5')

        test_predict = model.predict(resp_test)
        print(shape(test_predict))

        test_predict = test_predict.flatten()

        plt.plot(bcg_test[0], 'red')
        plt.plot(test_predict, 'blue')
        plt.title("Sine Wave with Smooth Random Amplitude Variation")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
