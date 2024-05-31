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
        a_zero = -0.0696744295499
        a_coefficients = [0.0985257910275, -0.143456940687, -0.0404705390332, 0.640995953202, 0.55963764938, -0.61783182995, -0.0340051552522, -0.138261808821, 0.00285513203664]
        b_coefficients = [0.0032569062913, 0.0103412471102, -0.330663690185, -0.360518437189, 0.855458124341, 0.34569161862, -0.103010876438, 0.0445445234745, -0.0184737838949]
        period = 60/self.heart_rate
        smoothness = 20

        respiration_signal_length = int(len(self.respiration) / self.sensor_frequency)
        x = np.linspace(0, respiration_signal_length, self.sensor_frequency * respiration_signal_length)
        noise = np.random.normal(0, 0.9, len(x))
        smoothed_noise = gaussian_filter1d(noise, smoothness)
        amplitude_envelope = smoothed_noise

        in_sum = []
        for n in range(9):
            in_sum.append(a_coefficients[n] * np.cos((2 * np.pi) / period * n * x) + b_coefficients[n] * np.sin((2 * np.pi) / period * n * x))

        wave = []
        for i in range(len(in_sum[0])):
            values = []
            for n in range(9):
                values.append(in_sum[n][i])
            wave.append(amplitude_envelope[i] + a_zero/2 + np.sum(values))

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

