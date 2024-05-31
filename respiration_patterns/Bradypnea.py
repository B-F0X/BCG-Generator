import random

from respiration_patterns.RespirationPattern import RespirationPattern


class Bradypnea(RespirationPattern):

    def __init__(self, respiration_rate=None, time=60, smoothness=200, amplitude=1.0, sensor_frequency=1000, start=0):
        if respiration_rate is None:
            respiration_rate = random.randint(7,10)
        super().__init__(respiration_rate, time, smoothness, amplitude, sensor_frequency, start)
