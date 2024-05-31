import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from respiration_patterns.Apnea import Apnea

matplotlib.use('TkAgg')

from respiration_patterns.CheyneStoke import CheyneStoke
from respiration_patterns.RespirationPattern import RespirationPattern
from wave_connector.WaveConnector import WaveConnector

matplotlib.use('TkAgg')


class CheyneStokes(RespirationPattern):
    def __init__(self, respiration_rate=15, smoothness=200, amplitude=1.0, sensor_frequency=1000,
                 length_of_cheyne_stokes=40, length_of_apnea=20, number_of_cheyen_stokes=2):
        self.length_of_cheyne_stokes = length_of_cheyne_stokes
        self.length_of_apnea = length_of_apnea
        self.number_of_cheyen_stokes = number_of_cheyen_stokes
        super().__init__(respiration_rate, 0, smoothness, amplitude, sensor_frequency, 0)

    def get(self):
        waves = []
        wave_connector = WaveConnector()

        for i in range(self.number_of_cheyen_stokes):
            cheyne_stoke = CheyneStoke(time=self.length_of_cheyne_stokes)
            waves.append(cheyne_stoke)
            if i != self.number_of_cheyen_stokes - 1:
                apnea = Apnea(time=self.length_of_apnea)
                waves.append(apnea)
        return wave_connector.connect(waves)

