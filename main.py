import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from bcg_generator.BcgGenerator import BcgGenerator
from cnn.CNNModel import CNNModel
from data_extractor.DataExtractor import DataExtractor
from lstm.LSTMModel import LSTMModel
from respiration_patterns.CheyneStoke import CheyneStoke
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

def create_heart_beats():
    kussmaul = Kussmaul(time=20)
    bradypnea = Bradypnea(time=2)
    apnea = Apnea(time=4)
    cheyne_strokes = CheyneStokes()
    wave_connector = WaveConnector()
    #connected_waves = wave_connector.connect([bradypnea, kussmaul])
    bcg = BcgGenerator(apnea.get(), 80, 1000)
    bcg.show()


def show_respiration_pattern1():
    data_extractor = DataExtractor()
    a0_4_rr19 = Normal(amplitude=0.5, respiration_rate=19, time=190)
    data_extractor.show_respiration_pattern('a0_4-rr19.csv')
    a0_4_rr19.show()
    data_extractor.save_respiration_pattern(a0_4_rr19, 'a0_4-rr19')


def show_respiration_pattern2():
    data_extractor = DataExtractor()
    wave_connector = WaveConnector()
    a2_7_rr12_1 = Normal(amplitude=2.8, respiration_rate=12, time=29)
    a1_3_rr21 = Normal(amplitude=1.3, respiration_rate=21, time=34)
    a2_7_rr12_2 = Normal(amplitude=2.7, respiration_rate=12, time=19)
    a2_7_rr12_a1_3_rr21_a2_7_rr12 = wave_connector.connect([a2_7_rr12_1, a1_3_rr21, a2_7_rr12_2])
    data_extractor.show_respiration_pattern('a2_7-rr12=a1_3-rr21=a2_7-rr12.csv')
    time_in_sec = int(len(a2_7_rr12_a1_3_rr21_a2_7_rr12) / 1000)
    t1 = np.linspace(0, time_in_sec, len(a2_7_rr12_a1_3_rr21_a2_7_rr12))
    plt.plot(t1, a2_7_rr12_a1_3_rr21_a2_7_rr12)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    data_extractor.save_connected_wave(a2_7_rr12_a1_3_rr21_a2_7_rr12, 'a2_7-rr12=a1_3-rr21=a2_7-rr12')


def show_respiration_pattern3():
    data_extractor = DataExtractor()
    wave_connector = WaveConnector()
    a1_1_rr11 = Normal(amplitude=1.1, respiration_rate=11, time=61)
    a0_6_rr24 = Normal(amplitude=0.6, respiration_rate=24, time=41)
    a1_1_rr11_a0_6_rr24 = wave_connector.connect([a1_1_rr11, a0_6_rr24])
    data_extractor.show_respiration_pattern('a1_1-rr11=a0_6-rr24.csv')
    time_in_sec = int(len(a1_1_rr11_a0_6_rr24) / 1000)
    t1 = np.linspace(0, time_in_sec, len(a1_1_rr11_a0_6_rr24))
    plt.plot(t1, a1_1_rr11_a0_6_rr24)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    data_extractor.save_connected_wave(a1_1_rr11_a0_6_rr24, 'a1_1-rr11=a0_6-rr24')


def show_respiration_pattern4():
    data_extractor = DataExtractor()
    a1_rr18 = Normal(amplitude=1, respiration_rate=18, time=108)
    data_extractor.show_respiration_pattern('a1-rr18.csv')
    a1_rr18.show()
    data_extractor.save_respiration_pattern(a1_rr18, 'a1-rr18')


def show_respiration_pattern5():
    data_extractor = DataExtractor()
    a2_5_rr9 = Normal(amplitude=2.5, respiration_rate=9, time=79)
    data_extractor.show_respiration_pattern('a2_5-rr9.csv')
    a2_5_rr9.show()
    data_extractor.save_respiration_pattern(a2_5_rr9, 'a2_5-rr9')


def show_respiration_pattern6():
    data_extractor = DataExtractor()
    wave_connector = WaveConnector()
    a0_8_rr18 = Normal(amplitude=0.8, respiration_rate=18, time=91)
    a1_4_rr14 = Normal(amplitude=1.4, respiration_rate=14, time=25)
    a0_8_rr18_a1_4_rr14 = wave_connector.connect([a0_8_rr18, a1_4_rr14])
    data_extractor.show_respiration_pattern('a0_8-rr18=a1_4-rr14.csv')
    time_in_sec = int(len(a0_8_rr18_a1_4_rr14) / 1000)
    t1 = np.linspace(0, time_in_sec, len(a0_8_rr18_a1_4_rr14))
    plt.plot(t1, a0_8_rr18_a1_4_rr14)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    data_extractor.save_connected_wave(a0_8_rr18_a1_4_rr14, 'a0_8-rr18=a1_4-rr14')


def show_respiration_pattern7():
    data_extractor = DataExtractor()
    a0_3_rr14 = Normal(amplitude=0.3, respiration_rate=14, time=199)
    data_extractor.show_respiration_pattern('a0_3-rr14.csv')
    a0_3_rr14.show()
    data_extractor.save_respiration_pattern(a0_3_rr14, 'a0_3-rr14')


def show_respiration_pattern8():
    data_extractor = DataExtractor()
    wave_connector = WaveConnector()
    a0_8_rr18_1 = Normal(amplitude=0.8, respiration_rate=18, time=18)
    a0_5_rr24_1 = Normal(amplitude=0.5, respiration_rate=24, time=41)
    a0_8_rr18_2 = Normal(amplitude=0.8, respiration_rate=18, time=32)
    a0_5_rr24_2 = Normal(amplitude=0.5, respiration_rate=24, time=46)
    a0_8_rr18_a0_5_rr24_a0_8_rr18_a0_5_rr24 = wave_connector.connect([a0_8_rr18_1, a0_5_rr24_1, a0_8_rr18_2, a0_5_rr24_2])
    data_extractor.show_respiration_pattern('a0_8-rr18=a0_5-rr24=a0_8-rr18=a0_5-rr24.csv')
    time_in_sec = int(len(a0_8_rr18_a0_5_rr24_a0_8_rr18_a0_5_rr24) / 1000)
    t1 = np.linspace(0, time_in_sec, len(a0_8_rr18_a0_5_rr24_a0_8_rr18_a0_5_rr24))
    plt.plot(t1, a0_8_rr18_a0_5_rr24_a0_8_rr18_a0_5_rr24)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    data_extractor.save_connected_wave(a0_8_rr18_a0_5_rr24_a0_8_rr18_a0_5_rr24, 'a0_8-rr18=a0_5-rr24=a0_8-rr18=a0_5-rr24')


def show_respiration_pattern9():
    data_extractor = DataExtractor()
    a0_8_rr12 = Normal(amplitude=0.8, respiration_rate=12, time=184)
    data_extractor.show_respiration_pattern('a0_8-rr12.csv')
    a0_8_rr12.show()
    data_extractor.save_respiration_pattern(a0_8_rr12, 'a0_8-rr12')


def show_respiration_pattern10():
    data_extractor = DataExtractor()
    a0_6_rr8 = Normal(amplitude=0.6, respiration_rate=8, time=130)
    data_extractor.show_respiration_pattern('a0_6-rr8.csv')
    a0_6_rr8.show()
    data_extractor.save_respiration_pattern(a0_6_rr8, 'a0_6-rr8')

def show_respiration_pattern11():
    data_extractor = DataExtractor()
    wave_connector = WaveConnector()
    apnea_1 = Apnea(time=14)
    cheyne = CheyneStoke(amplitude=3, respiration_rate=10, time=47)
    apnea_2 = Apnea(time=17)
    cheyne_stoke = wave_connector.connect([apnea_1, cheyne, apnea_2])
    data_extractor.show_respiration_pattern('cheyne_stoke.csv')
    time_in_sec = int(len(cheyne_stoke) / 1000)
    t1 = np.linspace(0, time_in_sec, len(cheyne_stoke))
    plt.plot(t1, cheyne_stoke)
    plt.title("Sine Wave with Smooth Random Amplitude Variation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    data_extractor.save_connected_wave(cheyne_stoke, 'cheyne_stoke')


if __name__ == '__main__':
    # create_heart_beats()
    # predict_with_lstm()
    # show_connected_waves()
    show_respiration_pattern11()


