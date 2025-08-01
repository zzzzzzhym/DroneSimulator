import os
import yaml
import numpy as np
import pandas as pd
import pickle


class Logger:
    """When the simulation logs data, it will be stored in a buffer. The buffer is a dictionary with keys defined in logger_config.py.
    The values in the buffer are lists which can be directly appended without copying the whole list. 
    After the simulation, the buffer is converted to a numpy array and stored in the output dictionary.
    """
    def __init__(self) -> None:
        self.config = self.load_config("logger_config.yaml")
        self.buffer = self.initialize_buffer()
        self.output = {}

    @staticmethod
    def load_config(filename: str) -> dict:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, filename)
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def initialize_buffer(self) -> dict:
        buffer = {}
        for key in self.config:
            buffer[key] = []
        return buffer
    
    def convert_buffer_to_output(self):
        """Assume buffer and the final result dict has the same keys"""
        for key in self.buffer:
            self.output[key] = np.array(self.buffer[key])

    def get_names_of_items_to_csv(self):
        items = []
        for key, val in self.config.items():
            if "can_save_to_file" in val:
                if val["can_save_to_file"]:
                    items.append(key)
        return items

    def make_data_frame(self) -> pd.DataFrame:
        df = pd.DataFrame()
        items = self.get_names_of_items_to_csv()
        for key in items:
            df[key] = self.output[key].tolist()
        return df

    def log_sim_result(self, file_name: str, type: str) -> None:
        if type == 'csv':
            suffix = ".csv"
        elif type == 'pkl':
            suffix = ".pkl"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(current_dir)
        file_path = os.path.join(upper_dir, "data", "training", file_name + suffix)
        if not os.path.exists(file_path):
            if type == 'csv':
                df = self.make_data_frame()
                df.to_csv(file_path, index=False, float_format='%.17f')
            elif type == 'pkl':
                with open(file_path, "wb") as f:
                    pickle.dump(self.buffer, f)
            print("Sim data is written into:\n" + os.path.relpath(file_path, os.getcwd()))
        else:
            raise ValueError("File already exist:\n" + file_path)

    def generate_column_map(self) -> dict:
        """Generate a dictionary that maps the keys in the buffer to their corresponding components.
        This is used to generate the header for the CSV file."""
        headers = self.get_names_of_items_to_csv()
        header_map = {header: idx for idx, header in enumerate(headers)}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        map_file_path = os.path.join(current_dir, "column_map.yaml")
        with open(map_file_path, "w") as f:
            for key, value in header_map.items():
                yaml.dump({key: value}, f)

