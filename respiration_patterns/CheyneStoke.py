import numpy as np
import matplotlib
from scipy.ndimage import gaussian_filter1d

from respiration_patterns.RespirationPattern import RespirationPattern
from wave_connector.WaveConnector import WaveConnector

matplotlib.use('TkAgg')


class CheyneStoke(RespirationPattern):
    def __init__(self, respiration_rate=15, time=40, smoothness=200, amplitude=1.0, sensor_frequency=1000, start=0):
        super().__init__(respiration_rate, time, smoothness, amplitude, sensor_frequency, start)

    def get(self):
        # Shift of the Sine wave
        phi = 0

        # Generate Gaussian noise with standard deviation of 0.9
        noise = np.random.normal(0, 0.9, len(self.x))

        # Apply a Gaussian filter to smooth the noise
        smoothed_noise = gaussian_filter1d(noise, self.smoothness)

        # Create the base amplitude with some offset to avoid negative values
        amplitude_envelope = 1 + smoothed_noise

        # Calculate the value which determines the length of the cheyne stoke based on the desired length in seconds
        b = 0.1 * (10 / self.time)

        # Create the sine wave with smooth random amplitude variation | amplitude_envelope * self.amplitude *
        return amplitude_envelope * self.amplitude * \
               np.sin(self.respiration_rate * np.pi * self.x + phi) * \
               np.sin(self.x * b * np.pi)


