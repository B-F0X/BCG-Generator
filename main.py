import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from respiration_patterns.Apnea import Apnea
from respiration_patterns.Bradypnea import Bradypnea
from respiration_patterns.CheyneStoke import CheyneStoke
from respiration_patterns.CheyneStokes import CheyneStokes
from respiration_patterns.Kussmaul import Kussmaul
from respiration_patterns.Tachypnea import Tachypnea

matplotlib.use('TkAgg')
from respiration_patterns.Hyperpnea import Hyperpnea
from respiration_patterns.Normal import Normal
from wave_connector.WaveConnector import WaveConnector


def show_connected_waves():
    wave_1 = Normal(time=15)
    wave_2 = Apnea(time=15)
    wave_3 = Bradypnea(time=30)
    cheyne_strokes = CheyneStokes()
    kussmaul = Kussmaul(time=10)
    wave_connector = WaveConnector()
    connected_waves = wave_connector.connect([kussmaul, cheyne_strokes])

    time_in_sec = int(len(connected_waves) / 1000 * 60)
    t1 = np.linspace(0, time_in_sec, len(connected_waves))
    print(len(connected_waves))
    plt.plot(t1, connected_waves)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    show_connected_waves()


