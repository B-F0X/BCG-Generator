import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


class BcgGenerator:
    def __init__(self, respiration, heart_rate, sensor_frequency):
        self.respiration = respiration
        self.heart_rate = heart_rate
        self.sensor_frequency = sensor_frequency
        self.bcg = []
        self.generate_bcg()

    def generate_bcg(self):
        # a_zero = -0.0696744295499
        # a_coefficients = [0.0985257910275, -0.143456940687, -0.0404705390332, 0.640995953202, 0.55963764938, -0.61783182995, -0.0340051552522, -0.138261808821, 0.00285513203664]
        # b_coefficients = [0.0032569062913, 0.0103412471102, -0.330663690185, -0.360518437189, 0.855458124341, 0.34569161862, -0.103010876438, 0.0445445234745, -0.0184737838949]
        # a_zero = -0.145420777778
        # a_coefficients = [0.256163009011, -0.223345092634, -0.20353043744, 0.95143882242, 0.204013830427, -0.491205811993, -0.0335306606699, -0.130482658195, -0.0389531548322, -0.00230857612643]
        # b_coefficients = [-0.00419294336209, 0.198857327096, -0.501330749241, -0.330685368354, 1.06845059818, 0.000782323196096, 0.0949406175667, -0.0600931696916, 0.00868709165913, -0.0324126924329]
        a_zero = -0.302552597222
        a_coefficients = [0.351796251469, -0.187878430323, -0.329835486088, 1.06244746604, 0.184064527731, -0.552099081135, 0.0403373057349, -0.162507365652, -0.0504072641987, 0.0202691412601]
        b_coefficients = [-0.126159090669, 0.343249061667, -0.558358746536, -0.388720331696, 1.1760922893, -0.0682864282916, 0.0906809468401, -0.0140154010527, -0.0288537930442, -0.0238021042076]
        period = 60/self.heart_rate
        smoothness = 20

        respiration_signal_length = int(len(self.respiration) / self.sensor_frequency)
        x = np.linspace(0, respiration_signal_length, self.sensor_frequency * respiration_signal_length)
        noise = np.random.normal(0, 0.9, len(x))
        smoothed_noise = gaussian_filter1d(noise, smoothness)
        amplitude_envelope = smoothed_noise

        in_sum = []
        for n in range(len(a_coefficients)):
            in_sum.append(a_coefficients[n] * np.cos((2 * np.pi) / period * n * x) + b_coefficients[n] * np.sin((2 * np.pi) / period * n * x))

        wave = []
        for i in range(len(in_sum[0])):
            values = []
            for n in range(len(a_coefficients)):
                values.append(in_sum[n][i])
            wave.append(amplitude_envelope[i] + a_zero/2 + np.sum(values))

        # noise2 = np.random.default_rng().uniform(-0.15,0.15,len(x))
        combined_wave = []
        for i in range(len(wave)):
            combined_wave.append(wave[i] + self.respiration[i])

        self.bcg = combined_wave

    def get_bcg(self):
        return self.bcg

    def show(self):
        respiration_signal_length = int(len(self.respiration) / self.sensor_frequency)
        x = np.linspace(0, respiration_signal_length, self.sensor_frequency * respiration_signal_length)
        plt.plot(x, self.bcg)
        plt.title("Generated BCG")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

