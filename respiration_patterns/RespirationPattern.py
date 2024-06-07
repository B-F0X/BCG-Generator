import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
matplotlib.use('TkAgg')


class RespirationPattern:

    def __init__(self, respiration_rate, time, smoothness, amplitude, sensor_frequency, start):
        # time_in_minutes = time / 60
        self.x = np.linspace(start, start + time, int(sensor_frequency * time))
        self.respiration_rate = respiration_rate / 30
        self.time = time
        self.smoothness = smoothness
        self.amplitude = amplitude
        self.sensor_frequency = sensor_frequency
        self.start = start
        self.get()

    def get(self):

        # Shift of the Sine wave
        phi = 0

        # Generate Gaussian noise with standard deviation of 0.9
        noise = np.random.normal(0, 6, len(self.x))

        # Apply a Gaussian filter to smooth the noise
        smoothed_noise = gaussian_filter1d(noise, self.smoothness)

        # Create the base amplitude with some offset to avoid negative values
        amplitude_envelope = 1 + smoothed_noise

        # Create the sine wave with smooth random amplitude variation
        return amplitude_envelope * self.amplitude * np.sin(self.respiration_rate * np.pi * self.x + phi)

    def get_zero_distance(self):
        return int((1 / self.respiration_rate) * self.sensor_frequency)

    def show(self):
        wave = self.get()
        time_in_sec = int(len(wave) / 1000 * 60)
        time = np.linspace(0, time_in_sec, len(wave))
        plt.plot(time, wave)
        plt.title("Sine Wave with Smooth Random Amplitude Variation")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
