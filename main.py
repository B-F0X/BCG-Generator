import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from cnn.CNNModel import CNNModel
from data_extractor.DataExtractor import DataExtractor
from lstm.LSTMModel import LSTMModel
from variables import variables

from respiration_patterns.Apnea import Apnea
from respiration_patterns.Bradypnea import Bradypnea
from respiration_patterns.CheyneStokes import CheyneStokes
from respiration_patterns.Kussmaul import Kussmaul
from respiration_patterns.Tachypnea import Tachypnea

matplotlib.use('TkAgg')
from respiration_patterns.Hyperpnea import Hyperpnea
from respiration_patterns.Normal import Normal
from wave_connector.WaveConnector import WaveConnector


def show_connected_waves():
    #normal = Normal(time=15)
    #apnea = Apnea(time=15)
    bradypnea = Bradypnea(time=30)
    #cheyne_strokes = CheyneStokes()
    #kussmaul = Kussmaul(time=10)
    tachypnea = Tachypnea(time=30)
    #hyperpnea = Hyperpnea(time=15)

    wave_connector = WaveConnector()
    connected_waves = wave_connector.connect([bradypnea, tachypnea])

    time_in_sec = int(len(connected_waves) / 1000 * 60)
    t1 = np.linspace(0, time_in_sec, len(connected_waves))
    print(len(connected_waves))
    plt.plot(t1, connected_waves)
    plt.ylim(-2.5, 2.5)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

def show_one_wave():
    waves = []
    waves.append(Normal(time=30))
    waves.append(Apnea(time=30))
    waves.append(Bradypnea(time=30))
    waves.append(CheyneStokes(length_of_cheyne_stokes=20, length_of_apnea=10))
    waves.append(Kussmaul(time=30))
    waves.append(Tachypnea(time=30))
    waves.append(Hyperpnea(time=30))

    Bradypnea(time=30).show()

    #wave_connector = WaveConnector()
    #connected_waves = wave_connector.connect([normal, apnea, bradypnea, cheyne_strokes, kussmaul, tachypnea, hyperpnea])

    for wave in waves:
        wave.show()

def show_data():
    respiration = pd.read_csv(variables.path_to_respiration_data + 'resp1.csv')
    bcg = pd.read_csv(variables.path_to_bottom_right_bcg + 'bcg_bottom_right1.csv')

    a = (max(respiration['data']) - min(respiration['data'])) / (max(bcg['data']) - min(bcg['data']))
    avg_respiration = (max(respiration['data']) + min(respiration['data'])) / 2
    avg_bcg = (max(bcg['data']) + min(bcg['data'])) / 2
    b = avg_respiration - a * avg_bcg

    for i in range(1, 40000):
        bcg['data'][i] = a * bcg['data'][i] + b

    plt.plot(bcg[:40000], 'red')
    plt.plot(respiration[:40000], 'green')
    plt.title('Plot of Values from CSV')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()


def predict_with_lstm():
    data_extractor = DataExtractor()
    data_extractor.get_data_from_combined_files(variables.path_to_combined_files)
    #lstm = LSTMModel(data_extractor)
    #lstm.run()
    cnn = CNNModel(data_extractor)
    cnn.run()


if __name__ == '__main__':
    predict_with_lstm()
    # show_connected_waves()


