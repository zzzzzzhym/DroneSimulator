import os
import yaml
import numpy as np


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
            # for scalar values, reshape to 2D to prevent hstack error in construct_csv_array
            if self.output[key].ndim == 1:  
                self.output[key] = self.output[key].reshape(-1, 1)

    def get_items_to_csv(self):
        items = []
        headers = []
        for key, val in self.config.items():
            if "can_save_to_file" in val:
                if val["can_save_to_file"]:
                    items.append(key)
                    if len(val["components"]) == 0:
                        headers.append(key)
                    else:
                        for suffix in val["components"]:
                            headers.append(key + "_" + suffix)
        return items, headers

    def construct_csv_array(self) -> tuple[str, np.ndarray]:
        items, headers = self.get_items_to_csv()
        first_line = ", ".join(headers)
        data = np.hstack([self.output[key] for key in items])
        return first_line, data

    def log_sim_result(self, file_name: str) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        upper_dir = os.path.dirname(current_dir)
        file_path = os.path.join(upper_dir, "data", "training", file_name+".csv")
        if not os.path.exists(file_path):
            first_line, array = self.construct_csv_array()
            np.savetxt(file_path, array, delimiter=',', fmt='%.17f', header=first_line, comments='')
            print("Sim data is written into:\n" + os.path.relpath(file_path, os.getcwd()))
        else:
            raise ValueError("File already exist:\n" + file_path)

    def generate_data_map(self) -> dict:
        """Generate a dictionary that maps the keys in the buffer to their corresponding components.
        This is used to generate the header for the CSV file."""
        _, headers = self.get_items_to_csv()
        header_map = {header: idx for idx, header in enumerate(headers)}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        map_file_path = os.path.join(current_dir, "data_map.yaml")
        with open(map_file_path, "w") as f:
            for key, value in header_map.items():
                yaml.dump({key: value}, f)

