import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import numpy as np
import pickle
import os

import model_config as config

class LearningDataset(Dataset):

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, conditions: np.ndarray) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
            conditions (np.ndarray): conditions
        """
        self.inputs = inputs
        self.outputs = outputs
        self.c = conditions

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx, :]
        output = self.outputs[idx, :]
        sample = {'input': torch.tensor(input), 
                  'output': torch.tensor(output), 
                  'c': torch.tensor(self.c)}
        return sample

def load_sim_data(file_name: str) -> np.ndarray:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "data")
    file_path = os.path.join(file_path, "training")    
    file_path = os.path.join(file_path, file_name)    
    if os.path.exists(file_path):
        # Load the data from the CSV file, skipping the first row (header)
        sim_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        return sim_data
    else:
        raise ValueError("No such file: " + file_path)

def convert_sim_to_training_data(sim_data: np.ndarray, label: int) -> LearningDataset:
    input = sim_data[:, 0:(3+4+4)]
    output = sim_data[:, (3+4+4):] # disturbance force
    result = LearningDataset(input, output, label)
    return result

def get_data_loaders(training_data: LearningDataset) -> tuple[DataLoader, DataLoader]:
    length = len(training_data)
    part1 = int(length*2/3)
    part2 = length - int(length*2/3)
    phi_set, a_set = random_split(training_data, [part1, part2])
    phi_loader = torch.utils.data.DataLoader(phi_set, batch_size=config.training['phi_shot'], shuffle=True)
    a_loader = torch.utils.data.DataLoader(a_set, batch_size=config.training['a_shot'], shuffle=True)
    return phi_loader, a_loader

def prepare_loader_sets(menu: list) -> tuple[list, list]:
    phi_set = []
    a_set = []
    condition_id = 0    # ID value doesn't matter
    for file in menu:
        data = load_sim_data(file)
        data = convert_sim_to_training_data(data, condition_id)
        phi_loader, a_loader = get_data_loaders(data)
        condition_id += 1
        phi_set.append(phi_loader)
        a_set.append(a_loader)
    return phi_set, a_set

if __name__ == "__main__":
    phi_set, a_set = prepare_loader_sets(["test_sample.csv"])
    for i, batch in enumerate(phi_set[0]):
        print((batch.keys()))

