import random

from respiration_patterns.RespirationPattern import RespirationPattern


class Hyperpnea(RespirationPattern):

    def __init__(self, respiration_rate=None, time=60, smoothness=200, amplitude=None, sensor_frequency=1000, start=0):
        if respiration_rate is None:
            respiration_rate = random.randint(22,29)
        if amplitude is None:
            amplitude = random.uniform(1.3, 2.3)
        super().__init__(respiration_rate, time, smoothness, amplitude, sensor_frequency, start)
