import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import numpy as np
import os
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt

import model_config


class LearningDataset(Dataset):

    def __init__(self, input: np.ndarray, output: np.ndarray, conditions: np.ndarray) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
            conditions (np.ndarray): conditions
        """
        self.input = input
        self.output = output
        self.c = conditions
        self.normalize_data()

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx, :]
        output = self.output[idx, :]
        sample = {'input': torch.tensor(input), 
                  'output': torch.tensor(output), 
                  'c': torch.tensor(self.c)}
        return sample
    
    def normalize_data(self):
        self.input[:, 7:11] *= 1
        self.input[:, 3:7] *= 1
        self.input[:, 0:3] *= 1

        self.output[:,0:3] *= 10

    def plot_data(self):
        fig, axs = plt.subplots(4,1)
        axs[0].plot(self.input[:, 0])
        axs[0].plot(self.input[:, 1])
        axs[0].plot(self.input[:, 2])
        axs[0].legend(['vx', 'vy', 'vz'])
        axs[1].plot(self.input[:, 3])
        axs[1].plot(self.input[:, 4])
        axs[1].plot(self.input[:, 5])
        axs[1].plot(self.input[:, 6])
        axs[1].legend(['q0', 'q1', 'q2', 'q3'])
        axs[2].plot(self.input[:, 7])
        axs[2].plot(self.input[:, 8])
        axs[2].plot(self.input[:, 9])
        axs[2].plot(self.input[:, 10])
        axs[2].legend(['F_motor_1', 'F_motor_1', 'F_motor_2', 'F_motor_3'])
        axs[3].plot(self.output[:, 0])
        axs[3].plot(self.output[:, 1])
        axs[3].plot(self.output[:, 2])
        axs[3].legend(['F_disturb_x', 'F_disturb_y', 'F_disturb_z'])


def load_sim_data(file_name: str) -> np.ndarray:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "data", "training", file_name)
    if os.path.exists(file_path):
        # Load the data from the CSV file, skipping the first row (header)
        sim_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return sim_data
    else:
        raise ValueError("No such file: " + file_path)

def convert_sim_to_training_data(sim_data: np.ndarray, label: int) -> LearningDataset:
    v = sim_data[:, 0:3]    # velocity
    q = sim_data[:, 3:7]    # quaternion
    f = sim_data[:, 7:11]   # motor force
    fd = sim_data[:, 11:]   # disturbance force
    input = np.hstack([v, q, f])
    result = LearningDataset(input, fd, label)
    return result

def get_data_loaders(training_data: LearningDataset, config: model_config.ModelConfig.Trainer) -> tuple[DataLoader, DataLoader]:
    length = len(training_data)
    part1 = int(length*2/3)
    part2 = length - int(length*2/3)
    phi_set, a_set = random_split(training_data, [part1, part2])
    phi_loader = torch.utils.data.DataLoader(phi_set, batch_size=config.phi_shot, shuffle=True)
    a_loader = torch.utils.data.DataLoader(a_set, batch_size=config.a_shot, shuffle=True)
    return phi_loader, a_loader

def prepare_datasets(menu: list) -> list[LearningDataset]:
    datasets = []
    condition_id = 0    # ID value doesn't matter
    for file in menu:
        data = load_sim_data(file)
        dataset = convert_sim_to_training_data(data, condition_id)
        condition_id += 1
        datasets.append(dataset)
    return datasets

def prepare_loadersets(datasets: list, config: model_config.ModelConfig.Trainer) -> tuple[list, list]:
    phi_set = []
    a_set = []
    for data in datasets:
        phi_loader, a_loader = get_data_loaders(data, config)
        phi_set.append(phi_loader)
        a_set.append(a_loader)
    return phi_set, a_set

def prepare_back2back_datasets(menu: list) -> list[LearningDataset]:
    datasets = []
    condition_id = 0    # ID value doesn't matter
    for file in menu:
        data = load_back2back_data(file)
        dataset = convert_sim_to_training_data(data, condition_id)
        condition_id += 1
        datasets.append(dataset)
    return datasets

def load_back2back_data(file_name: str) -> np.ndarray:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "data", "training_back2back", file_name)
    if os.path.exists(file_path):
        # Load the csv using a pandas.DataFrame
        df = pd.read_csv(file_path)

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # Copy all the data to a dictionary, and make things np.ndarrays
        data = {}
        for field in df.columns[1:]:
            data[field] = np.array(df[field].tolist(), dtype=float)

        sim_data =np.hstack((data['v'], data['q'], data['pwm']/1000, data['fa']))
        
        return sim_data
    else:
        raise ValueError("No such file: " + file_path)



if __name__ == "__main__":
    data_list = ["test_air_drag_0.csv"]
    dataset: list[LearningDataset] = prepare_datasets(data_list)
    phi_set, a_set = prepare_loadersets(dataset, model_config.ModelConfig(len(data_list)).Trainer)
    for batch_idx, data_batch in enumerate(phi_set[0]):
        print(f"Batch {batch_idx + 1}:")
        print(f"Data:\n{data_batch}")
    # dataset = prepare_back2back_datasets(['custom_random3_baseline_10wind.csv',
    #                                       'custom_random3_baseline_20wind.csv',
    #                                       'custom_random3_baseline_30wind.csv',
    #                                       'custom_random3_baseline_40wind.csv',
    #                                       'custom_random3_baseline_50wind.csv',
    #                                       'custom_random3_baseline_nowind.csv'])

    for data in dataset:
        data.plot_data()
    plt.show()

