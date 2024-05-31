import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


class BcgGenerator:
    def __init__(self):
        return

    def show(self, respiration):
        a_zero = -0.0696744295499
        a_coefficients = [0.0985257910275, -0.143456940687, -0.0404705390332, 0.640995953202, 0.55963764938, -0.61783182995, -0.0340051552522, -0.138261808821, 0.00285513203664]
        b_coefficients = [0.0032569062913, 0.0103412471102, -0.330663690185, -0.360518437189, 0.855458124341, 0.34569161862, -0.103010876438, 0.0445445234745, -0.0184737838949]
        period = 0.5
        smoothness = 20
        x = np.linspace(0, 10, 1000 * 10)
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

        plt.plot(x, wave)
        plt.title("Sine Wave with Smooth Random Amplitude Variation")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

        respiration.show()

        respiration_wave = respiration.get()
        combined_wave = []
        for i in range(len(wave)):
            combined_wave.append(wave[i] + respiration_wave[i])

        plt.plot(x, combined_wave)
        plt.title("Sine Wave with Smooth Random Amplitude Variation")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

