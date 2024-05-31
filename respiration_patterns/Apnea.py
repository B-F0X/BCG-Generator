import numpy as np
from scipy.ndimage import gaussian_filter1d
from respiration_patterns.RespirationPattern import RespirationPattern


class Apnea(RespirationPattern):

    def __init__(self, respiration_rate=0, time=60, smoothness=200, amplitude=0.0, sensor_frequency=1000, start=0):
        super().__init__(respiration_rate, time, smoothness, amplitude, sensor_frequency, start)

    def get(self):
        # Generate Gaussian noise with standard deviation of 0.2
        noise = np.random.normal(0, 0.2, len(self.x))

        # Apply a Gaussian filter to smooth the noise
        return gaussian_filter1d(noise, self.smoothness)

    def get_zero_distance(self):
        return 1
