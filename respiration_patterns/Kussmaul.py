from respiration_patterns.RespirationPattern import RespirationPattern
import random
from scipy import signal
import numpy as np
from scipy.ndimage import gaussian_filter1d

class Kussmaul(RespirationPattern):
    def __init__(self, respiration_rate=None, time=60, smoothness=200, amplitude=None, sensor_frequency=1000, start=0):
        if respiration_rate is None:
            respiration_rate = random.randint(22,29)
        if amplitude is None:
            amplitude = random.uniform(1.3, 2.3)
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

        # Create the sine wave with smooth random amplitude variation
        return amplitude_envelope * self.amplitude * signal.sawtooth(self.respiration_rate * np.pi * self.x, 0.8)
