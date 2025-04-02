import numpy as np
import os
import matplotlib.pyplot as plt
import yaml

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

import normalization

class LearningDataset(Dataset):

    def __init__(self, input: np.ndarray, output: np.ndarray, condition_id: int) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
            condition_id (np.ndarray): an ID number to label the condition of the data
        """
        self.input = input
        self.output = output
        self.c = condition_id

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx, :]
        output = self.output[idx, :]
        sample = {'input': torch.tensor(input), 
                  'output': torch.tensor(output), 
                  'c': torch.tensor(self.c)}
        return sample


class DataManager:
    """Data manager is responsible for:
    1. defining the list of data to be used to train
    2. check which columns are input and which are labels
    3. generate LearningDataset and then LoaderSets for trainer
    4. normalize the data"""
    def __init__(self, data_menu: list, input_label_map_file: str, column_map_file: str, can_skip_io_normalizaiton: bool) -> None:
        self.data_menu = data_menu
        self.input_label_map = DataManager.get_map(input_label_map_file)
        self.column_map = DataManager.get_map(column_map_file)
        self.input_columns, self.label_columns = DataManager.find_input_label_column(self.input_label_map, self.column_map)
        self.config = DataManager.load_config("data_manager_config.yaml")
        self.input_normalization: dict[int, normalization.Normalization] = {}
        self.label_normalization: dict[int, normalization.Normalization] = {}
        self.input_mean_vector, self.input_scale_vector, self.label_mean_vector, self.label_scale_vector = self.make_normalization_params(column_map_file, can_skip=can_skip_io_normalizaiton)

    @staticmethod
    def find_input_label_column(input_label_map: dict, column_map: dict):
        """use the config files to find the input and label columns"""
        input_columns = []
        for field in input_label_map["input"]:
            input_columns.append(column_map[field])
        label_columns = []
        for field in input_label_map["label"]:
            label_columns.append(column_map[field])    
        return input_columns, label_columns
    
    @staticmethod
    def get_map(map_file: str) -> dict:
        map_file_path = DataManager.get_path_to_data_file(map_file)
        with open(map_file_path, 'r') as file:
            map = yaml.safe_load(file)
        return map

    @staticmethod
    def get_path_to_data_file(file_name: str) -> str:
        """Get the path to the data file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(current_dir)
        file_path = os.path.join(upper_dir, "data", "training", file_name)
        return file_path

    @staticmethod
    def load_config(config_file: str):
        """Load configuration from YAML file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_file)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config    

    @staticmethod
    def load_sim_data(file_name: str) -> np.ndarray:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(os.path.dirname(current_dir), "data", "training", file_name)
        if os.path.exists(file_path):
            # Load the data from the CSV file, skipping the first row (header)
            sim_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            return sim_data
        else:
            raise ValueError("No such file: " + file_path)

    def convert_sim_to_training_data(self, sim_data: np.ndarray, condition_id: int) -> LearningDataset:
        # normalize data before putting it into the dataset
        result = LearningDataset((sim_data[:, self.input_columns] - self.input_mean_vector)*self.input_scale_vector, 
                                 (sim_data[:, self.label_columns] - self.label_mean_vector)*self.label_scale_vector,
                                 condition_id)
        return result

    def prepare_datasets(self, data_menu: list) -> list[LearningDataset]:
        datasets = []
        condition_id = 0    # ID value doesn't matter, but each different condition should have different ID
        for file in data_menu:
            data = DataManager.load_sim_data(file)
            dataset = self.convert_sim_to_training_data(data, condition_id)
            condition_id += 1
            datasets.append(dataset)
        return datasets

    def get_data_loaders(self, training_data: LearningDataset) -> tuple[DataLoader, DataLoader]:
        """Separate the dataset into two parts: part1 for phi NN and part2 for adaptation coeffecients, 
        generate a data loader for each part"""
        length = len(training_data)
        ratio = self.config["phi_shot"]/(self.config["phi_shot"] + self.config["a_shot"])
        part_phi = int(length*ratio)
        part_a = length - part_phi
        phi_set, a_set = random_split(training_data, [part_phi, part_a])
        phi_loader = torch.utils.data.DataLoader(phi_set, batch_size=self.config["phi_shot"], shuffle=True)
        a_loader = torch.utils.data.DataLoader(a_set, batch_size=self.config["a_shot"], shuffle=True)
        return phi_loader, a_loader

    def prepare_loadersets(self, datasets: list[LearningDataset]) -> tuple[list[DataLoader], list[DataLoader]]:
        """Generate a list of data loaders for phi and a respectively"""
        phi_set = []
        a_set = []
        for data in datasets:
            phi_loader, a_loader = self.get_data_loaders(data)
            phi_set.append(phi_loader)
            a_set.append(a_loader)
        return phi_set, a_set

    def get_data(self) -> tuple[list[DataLoader], list[DataLoader]]:
        """Main API of DataManager"""
        datasets = self.prepare_datasets(self.data_menu)
        phi_set, a_set = self.prepare_loadersets(datasets)
        return phi_set, a_set
    
    def normalize_data(self):
        """Normalize the data using the mean and standard deviation"""
        # initialization
        for column in self.input_columns:
            self.input_normalization[column] = normalization.Normalization()
        for column in self.label_columns:
            self.label_normalization[column] = normalization.Normalization()
        # traverse the datasets
        for file in self.data_menu:
            dataset = DataManager.load_sim_data(file)
            for column in self.input_columns:
                self.input_normalization[column].add_batch(dataset[:, column])
            for column in self.label_columns:
                self.label_normalization[column].add_batch(dataset[:, column])
    
    def generate_normalization_params_file(self, output_path: str) -> None:
        """Generate the normalization parameters for input and output"""
        content = {"input": {}, "output": {}}
        for column in self.input_columns:
            mean, scale = self.input_normalization[column].get_normalization_params()
            content["input"][column] = {"mean": mean, "scale": scale}
        for column in self.label_columns:
            mean, scale = self.label_normalization[column].get_normalization_params()
            content["output"][column] = {"mean": mean, "scale": scale}
        with open(output_path, 'w') as file:
            yaml.dump(content, file)
        
    def make_normalization_params(self, column_map_file: str, can_skip: bool) -> None:
        # if the normalization params file is not generated, generate it
        column_map_file_path = DataManager.get_path_to_data_file(column_map_file)
        column_map_dir = os.path.dirname(column_map_file_path)
        normalization_params_file_path = os.path.join(column_map_dir, "normalization_params.yaml")
        if not os.path.exists(normalization_params_file_path):
            print("Normalization params file not found, generating it and saving it to\n" + os.path.relpath(normalization_params_file_path))
            self.normalize_data()
            self.generate_normalization_params_file(normalization_params_file_path)
            print("Normalization params file generated")
        # load the normalization params file
        print("Loading normalization params file from\n" + os.path.relpath(normalization_params_file_path))
        with open(normalization_params_file_path, 'r') as file:
            normalization_params = yaml.safe_load(file)
        # make normalization matrix
        mean = []
        scale = []
        for columns in normalization_params["input"].keys():
            if columns in self.input_columns:
                mean.append(normalization_params["input"][columns]["mean"])
                scale.append(normalization_params["input"][columns]["scale"])
        input_mean_vector = np.array(mean)
        input_scale_vector = np.array(scale)
        mean = []
        scale = []
        for columns in normalization_params["output"].keys():
            if columns in self.label_columns:
                mean.append(normalization_params["output"][columns]["mean"])
                scale.append(normalization_params["output"][columns]["scale"])
        label_mean_vector = np.array(mean)
        label_scale_vector = np.array(scale)
        if can_skip:
            input_mean_vector = np.zeros(input_mean_vector.shape)
            input_scale_vector = np.ones(input_scale_vector.shape)
            label_mean_vector = np.zeros(label_mean_vector.shape)
            label_scale_vector = np.ones(label_scale_vector.shape)
        return input_mean_vector, input_scale_vector, label_mean_vector, label_scale_vector



