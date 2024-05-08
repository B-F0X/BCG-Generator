import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


class WaveConnector:

    def bezier_formula(self, t, start, control1, control2, end):
        # B(t) = (1-t)^3 * P0 + 3*(1-t)^2 * t * P1 + 3*(1-t) * t^2 * P2 + t^3 * P3
        return ((1 - t) ** 3 * start +
                3 * (1 - t) ** 2 * t * control1 +
                3 * (1 - t) * t ** 2 * control2 +
                t ** 3 * end)

    def bezier_curve(self, start, control1, control2, end, number_of_values):
        t_values = np.linspace(0, 1, number_of_values)
        curve_points = np.array([self.bezier_formula(t, start, control1, control2, end) for t in t_values])
        control_points = np.array([start, control1, control2, end])
        return curve_points, control_points

    def find_last_extreme(self, array, start):
        point = self.find_next_extreme(array[::-1], start)
        point[0] = len(array) - point[0] + start
        return point

    def find_next_extreme(self, array, start):
        i = 0
        if array[i] < array[i + 1]:
            while array[i] < array[i + 1]:
                i += 1
            return np.array([start + i, array[i]])

        while array[i] > array[i + 1]:
            i += 1
        return np.array([start + i, array[i]])

    def findIntersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        return [px, py]

    def connect(self, waves):
        if len(waves) < 2:
            print("Error: Two or more waves are required", file=sys.stderr)
            exit(0)

        wave_1 = waves[0].get()

        for i in range(1, len(waves)):
            wave_2 = waves[i].get()
            zero_distance_1 = waves[i-1].get_zero_distance()
            zero_distance_2 = waves[i].get_zero_distance()
            start_of_wave_2 = len(wave_1) + 1


            # Find the control points at the most recent extremes of both waves
            extreme_1 = self.find_last_extreme(wave_1, 0)
            extreme_2 = self.find_next_extreme(wave_2, start_of_wave_2)

            # Define the start as the end of wave one - one extreme
            start_x = len(wave_1) - zero_distance_1
            start_y = wave_1[start_x]
            start = np.array([start_x, start_y])

            # Define the end as the start of wave two + one extreme
            end_x = zero_distance_2 + start_of_wave_2 - 1
            end_y = wave_2[zero_distance_2]
            end = np.array([end_x, end_y])

            start_minus_one_y = wave_1[start_x - 1]
            end_plus_one_y = wave_2[end_x - start_of_wave_2 + 2]

            control_1 = np.array(self.findIntersection(start_x, start_y, start_x - 1, start_minus_one_y, extreme_1[0],
                                                  extreme_1[1], extreme_1[0], extreme_1[1] - 1))
            control_2 = np.array(self.findIntersection(end_x, end_y, end_x + 1, end_plus_one_y, extreme_2[0],
                                                  extreme_2[1], extreme_2[0], extreme_2[1] - 1))

            # if the connection is an arch, then move wave_1 and wave_2 closer together
            if (control_1[1] > start_y and control_2[1] > end_y) or (control_1[1] < start_y and control_2[1] < end_y):
                connection, control_points = self.bezier_curve(start, control_1, control_2, end, int((end_x - start_x + 1)/2))
            else:
                connection, control_points = self.bezier_curve(start, control_1, control_2, end, end_x - start_x + 1)

            # Cut end of wave 1 and start of wave 2
            wave_1_1 = wave_1[:-zero_distance_1]
            wave_2_2 = wave_2[zero_distance_2 + 1:]

            # Concatenate the waveforms to make a continuous signal
            wave_1 = np.concatenate([wave_1_1, connection[:, 1], wave_2_2])

        return wave_1
