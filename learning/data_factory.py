import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split

import normalization

class LearningDataset(Dataset):

    def __init__(self,
        input: np.ndarray,
        output: np.ndarray,
        condition_id: int,
        input_mean_vector,
        input_scale_vector,
        label_mean_vector,
        label_scale_vector,
        source_file: str) -> None:
        """

        Args:
            inputs (np.ndarray): velocity (3) + quaternion (4) + input (4) = 11
            outputs (np.ndarray): disturbance force (3)
            condition_id (np.ndarray): an ID number to label the condition of the data
        """
        self.input = input
        self.output = output
        self.c = condition_id
        self.source_file = source_file
        self.input_mean_vector = input_mean_vector
        self.input_scale_vector = input_scale_vector
        self.label_mean_vector = label_mean_vector
        self.label_scale_vector = label_scale_vector

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = self.input[idx, :]
        output = self.output[idx, :]
        sample = {'input': torch.tensor(input), 
                  'output': torch.tensor(output), 
                  'c': torch.tensor(self.c)}
        return sample


class DataFactory:
    """Data manager is responsible for:
    1. defining the list of data to be used to train
    2. check which columns are input and which are labels
    3. generate LearningDataset and then LoaderSets for trainer
    4. normalize the data"""
    def __init__(self, data_menu: list, input_label_map_file: str, column_map_file: str, can_skip_io_normalizaiton: bool) -> None:
        self.data_menu = data_menu
        self.input_label_map = DataFactory.get_map(input_label_map_file)
        self.column_map = DataFactory.get_map(column_map_file)
        self.input_columns, self.label_columns = DataFactory.find_input_label_column(self.input_label_map, self.column_map)
        self.config = DataFactory.load_config("data_factory_config.yaml")
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
        map_file_path = DataFactory.get_path_to_data_file(map_file)
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

    def convert_sim_to_training_data(self, sim_data: np.ndarray, condition_id: int, file: str) -> LearningDataset:
        # normalize data before putting it into the dataset
        result = LearningDataset((sim_data[:, self.input_columns] - self.input_mean_vector)*self.input_scale_vector, 
                                 (sim_data[:, self.label_columns] - self.label_mean_vector)*self.label_scale_vector,
                                 condition_id,
                                 self.input_mean_vector,
                                 self.input_scale_vector,
                                 self.label_mean_vector,
                                 self.label_scale_vector,
                                 file)
        return result

    def prepare_datasets(self, data_menu: list, can_inspect_data: bool=True) -> list[LearningDataset]:
        """Prepare the datasets from the data menu
        Each file in the data menu is a different condition, the length of the dataset is the number of conditions"""
        datasets = []
        condition_id = 0    # ID value doesn't matter, but each different condition should have different ID
        for file in data_menu:
            data = DataFactory.load_sim_data(file)
            if can_inspect_data:
                self.inspect_data(data, file)  # inspect the data 
            dataset = self.convert_sim_to_training_data(data, condition_id, file)
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
        """Main API of DataFactory"""
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
            dataset = DataFactory.load_sim_data(file)
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
        column_map_file_path = DataFactory.get_path_to_data_file(column_map_file)
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

    def inspect_data(self, sim_data, file: str) -> None:
        merged_columns = self.input_columns + self.label_columns
        merged_titles = self.input_label_map["input"] + self.input_label_map["label"]
        data = sim_data[:, merged_columns]
        DataFactory.plot_dataset_distribution_grid(data, merged_titles, file)
            
    @staticmethod
    def plot_dataset_distribution_grid(data: np.ndarray, sub_titles: list[str], title: str, bins=50, figsize=(20, 10), cols=7):
        num_features = data.shape[1]
        rows = (num_features + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        for i in range(num_features):
            sns.histplot(data[:, i], bins=bins, kde=True, ax=axes[i])
            axes[i].set_title(sub_titles[i])
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True)
        for j in range(num_features, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title)



