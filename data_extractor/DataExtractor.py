import math
from os import walk

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from variables import variables


class DataExtractor:
    def __init__(self):
        self.sensor_frequency = None
        self.respiration_data_folder = None
        self.bcg_data_folder = None
        self.respiration_data = None
        self.bcg_data = None

    def get_data(self):
        return self.respiration_data, self.bcg_data

    def get_data_from_different_files(self, sensor_frequency, respiration_data_folder, bcg_data_folder):
        self.sensor_frequency = sensor_frequency
        self.respiration_data_folder = respiration_data_folder
        self.bcg_data_folder = bcg_data_folder
        self.respiration_data = self.get_data_of_type(self.respiration_data_folder)
        self.bcg_data = self.get_data_of_type(self.bcg_data_folder)

    def get_data_of_type(self, data):
        downsampling_factor = int(1000 / self.sensor_frequency)
        values_per_minute = self.sensor_frequency * 60
        combined_data = []
        for data_folder in data:
            filenames = next(walk(data_folder), (None, None, []))[2]
            filenames.sort()
            data_of_folder = []
            for data_file in filenames:
                data_table = pd.read_csv(data_folder + data_file)
                data_array = data_table['data'].to_numpy()
                data_array = data_array[::downsampling_factor]
                data_array = np.array([[num] for num in data_array])
                scaler = MinMaxScaler(feature_range=(-1, 1))
                data_array = scaler.fit_transform(data_array)
                data_of_file = []
                i = 0
                while i + values_per_minute < len(data_array):
                    data_of_file.append(data_array[i:i + values_per_minute])
                    i += 200
                data_of_folder.append(data_of_file)
            combined_data.append(data_of_folder)
        return combined_data

    def get_data_from_combined_files(self, combined_files_folder):
        filenames = next(walk(combined_files_folder), (None, None, []))[2]
        filenames.sort()
        combined_respiration_data = []
        combined_bcg_data = []
        for data_file in filenames:
            data_table = pd.read_csv(combined_files_folder + data_file)
            resp = self.column_to_training_data(data_table, 'resp')
            bcg = self.column_to_training_data(data_table, 'bcg')
            combined_respiration_data.append(resp)
            combined_bcg_data.append(bcg)
        self.respiration_data = combined_respiration_data
        self.bcg_data = combined_bcg_data

    def column_to_training_data(self, data_table, column_name):
        values_per_minute = 3000
        data_array = data_table[column_name].to_numpy()
        data_array = np.array([[num] for num in data_array])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_array = scaler.fit_transform(data_array)
        data_of_file = []
        i = 0
        while i + values_per_minute < len(data_array):
            data_of_file.append(data_array[i:i + values_per_minute])
            i += 400
        return data_of_file

    def show_respiration_patterns(self):
        folder = variables.path_to_respiration_patterns
        filenames = next(walk(folder), (None, None, []))[2]
        for data_file in filenames:
            data_table = pd.read_csv(folder + data_file)
            data_array = data_table['data'].to_numpy()
            time_in_sec = int(len(data_array) / 1000)
            time = np.linspace(0, time_in_sec, len(data_array))
            print(f"{data_file}: {time_in_sec} seconds")
            plt.plot(time, data_array)
            plt.title(data_file)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()

    def show_respiration_pattern(self, filename):
        folder = variables.path_to_respiration_patterns
        filenames = next(walk(folder), (None, None, []))[2]
        data_table = pd.read_csv(folder + filename)
        data_array = data_table['data'].to_numpy()
        time_in_sec = int(len(data_array) / 1000)
        time = np.linspace(0, time_in_sec, len(data_array))
        print(f"{filename}: {time_in_sec} seconds")
        plt.plot(time, data_array)
        plt.title(filename)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()

    def save_respiration_pattern(self, data, name):
        folder = variables.path_to_created_respiration_patterns
        table = {'resp': data.get()}
        df = pd.DataFrame(table)
        df.to_csv(folder + name + '.csv', index=False)


    def save_connected_wave(self, data, name):
        folder = variables.path_to_created_respiration_patterns
        table = {'resp': data}
        df = pd.DataFrame(table)
        df.to_csv(folder + name + '.csv', index=False)

    ####### Helper Functions ##############

    def combine_signals_to_one_file(self):
        resp_data, filenames = self.get_signal_of_type(variables.path_to_respiration_data)
        bcg1_data, f = self.get_signal_of_type(variables.path_to_top_left_bcg)
        bcg2_data, f = self.get_signal_of_type(variables.path_to_bottom_left_bcg)
        bcg3_data, f = self.get_signal_of_type(variables.path_to_bottom_right_bcg)
        bcg4_data, f = self.get_signal_of_type(variables.path_to_top_right_bcg)
        for i in range(len(resp_data)):
            print("Combine date to file " + filenames[i])
            combined_file = {'resp': resp_data[i],
                             'bcg1': bcg1_data[i]}
            df = pd.DataFrame(combined_file)
            df.to_csv('data_extractor/combined_files/bcg1' + filenames[i], index=False)
            combined_file = {'resp': resp_data[i],
                             'bcg2': bcg2_data[i]}
            df = pd.DataFrame(combined_file)
            df.to_csv('data_extractor/combined_files/bcg2' + filenames[i], index=False)
            combined_file = {'resp': resp_data[i],
                             'bcg3': bcg3_data[i]}
            df = pd.DataFrame(combined_file)
            df.to_csv('data_extractor/combined_files/bcg3' + filenames[i], index=False)
            combined_file = {'resp': resp_data[i],
                             'bcg4': bcg4_data[i]}
            df = pd.DataFrame(combined_file)
            df.to_csv('data_extractor/combined_files/bcg4' + filenames[i], index=False)




    def get_signal_of_type(self, data_folder):
        filenames = next(walk(data_folder), (None, None, []))[2]
        filenames.sort()
        data_of_folder = []
        for data_file in filenames:
            data = pd.read_csv(data_folder + data_file)
            data_array = data['data'].to_numpy()
            data_array = data_array[::10]
            data_of_folder.append(data_array)
        return data_of_folder, filenames

    def remove_leading_zeros(self):
        data_folder = variables.path_to_edited_files
        filenames = next(walk(data_folder), (None, None, []))[2]
        filenames.sort()
        for data_file in filenames:
            data = pd.read_csv(data_folder + data_file)
            resp = data['resp']
            bcg = data[data_file[:4]]
            counter = 0
            if math.isnan(resp[0]):
                while math.isnan(resp[counter]):
                    counter += 1
                resp = resp[counter:].to_numpy()
                bcg = bcg[:-counter].to_numpy()

            if math.isnan(bcg[0]):
                while math.isnan(bcg[counter]):
                    counter += 1
                bcg = bcg[counter:].to_numpy()
                resp = resp[:-counter].to_numpy()

            combined_file = {'resp': resp,
                             'bcg': bcg}
            df = pd.DataFrame(combined_file)
            df.to_csv('data_extractor/filtered_files/' + data_file, index=False)
