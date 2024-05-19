import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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
    normal = Normal(time=15)
    apnea = Apnea(time=15)
    bradypnea = Bradypnea(time=15)
    cheyne_strokes = CheyneStokes()
    kussmaul = Kussmaul(time=10)
    tachypnea = Tachypnea(time=10)
    hyperpnea = Hyperpnea(time=15)

    wave_connector = WaveConnector()
    connected_waves = wave_connector.connect([normal, apnea, bradypnea, cheyne_strokes, kussmaul, tachypnea, hyperpnea])

    time_in_sec = int(len(connected_waves) / 1000 * 60)
    t1 = np.linspace(0, time_in_sec, len(connected_waves))
    print(len(connected_waves))
    plt.plot(t1, connected_waves)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


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
    lstm = LSTMModel()
    lstm.run()


if __name__ == '__main__':
    predict_with_lstm()


