import os
import torch

import model
import data_factory
import trainer
import validator

class TrainingManager:
    def __init__(self) -> None:
        self.data_factory_instance = None
        self.model_factory_instance = None
        self.trainer_instance = None        
        self.validator_instance = None        
        self.dim_of_input = None
        self.dim_of_label = None
        self.num_of_conditions = None

    def set_up_data_factory(self, data_menu: list, input_label_map_file: str, column_map_file: str, can_skip_io_normalizaiton: bool, can_inspect_data=False) -> None:
        print("Setting up data factory...")
        self.data_factory_instance = data_factory.DataFactory(
            data_menu, input_label_map_file, column_map_file, can_skip_io_normalizaiton
        )
        self.loaderset_phi, self.loaderset_a = self.data_factory_instance.get_data(can_inspect_data)
        self.dim_of_input = len(self.data_factory_instance.input_columns)
        self.dim_of_label = len(self.data_factory_instance.label_columns)
        self.num_of_conditions = len(self.data_factory_instance.data_menu) # assume each data file has a unique condition

    def set_up_model_factory(self) -> None:
        print("Setting up model factory...")
        self.model_factory_instance = model.ModelFactory(
            self.num_of_conditions,
            self.dim_of_input,
            self.data_factory_instance.input_mean_vector,
            self.data_factory_instance.input_scale_vector,
            self.data_factory_instance.label_mean_vector,
            self.data_factory_instance.label_scale_vector,
        )

    def set_up_trainer(self) -> None:
        print("Setting up trainer...")
        loaderset_phi, loaderset_a = self.data_factory_instance.get_data()
        phi_net, h_net = self.model_factory_instance.generate_nets()
        self.trainer_instance = trainer.Trainer(
            phi_net,
            h_net,
            loaderset_phi,
            loaderset_a,
            len(self.data_factory_instance.label_columns)
        )

    def set_up_validator(self, validation_data_menu, can_inspect_data) -> None:
        self.validator_instance = validator.Evaluator()
        self.validator_instance.load_model(self.trainer_instance.phi_net, self.trainer_instance.h_net)
        self.validator_instance.load_dataset(self.data_factory_instance.prepare_datasets(validation_data_menu, can_inspect_data))

    def set_up(self,
               data_menu: list,
               input_label_map_file: str,
               column_map_file: str,
               can_skip_io_normalizaiton: bool,
               can_inspect_data: bool,
               validation_data_menu: list
               ) -> None:
        self.set_up_data_factory(data_menu, input_label_map_file, column_map_file, can_skip_io_normalizaiton, can_inspect_data)
        self.set_up_model_factory()
        self.set_up_trainer()
        self.set_up_validator(validation_data_menu, can_inspect_data)

    def train(self) -> None:
        self.trainer_instance.train_model(self.validator_instance.callback_validation)

    def save_model(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "model", name + ".pth")
        torch.save({"phi": self.trainer_instance.phi_net.state_dict(),
                    "h": self.trainer_instance.h_net.state_dict(),
                    "model_factory_init_args": self.model_factory_instance.generate_self_config(), 
                    "phi_net_io_fields": self.data_factory_instance.input_label_map}, 
                    file_path)
        print(f"Model saved to {os.path.relpath(file_path, os.getcwd())}")


class TestManager:
    def __init__(self) -> None:
        self.data_factory_instance = None
        self.model_factory_instance = None
        self.trainer_instance = None        
        self.dim_of_input = None
        self.dim_of_label = None
        self.num_of_conditions = None

    def set_up(self,
               data_menu: list,
               input_label_map_file: str,
               column_map_file: str,
               can_skip_io_normalizaiton: bool,
               ) -> None:
        self.set_up_data_factory(data_menu, input_label_map_file, column_map_file, can_skip_io_normalizaiton)
        self.set_up_tester()

    def set_up_data_factory(self, data_menu: list, input_label_map_file: str, column_map_file: str, can_skip_io_normalizaiton: bool) -> None:
        self.data_factory_instance = data_factory.DataFactory(
            data_menu, input_label_map_file, column_map_file, can_skip_io_normalizaiton
        )
        self.dim_of_input = len(self.data_factory_instance.input_columns)
        self.dim_of_label = len(self.data_factory_instance.label_columns)
        self.num_of_conditions = len(self.data_factory_instance.data_menu) # assume each data file has a unique condition

    def set_up_tester(self) -> None:
        self.tester_instance = validator.Evaluator()

    def test(self, phi_net: model.MultilayerNet, h_net: model.MultilayerNet, data_menu_validation: list) -> None:
        datasets = self.data_factory_instance.prepare_datasets(data_menu_validation, False)        
        self.tester_instance.load_model(phi_net, h_net)
        self.tester_instance.load_dataset(datasets)
        self.tester_instance.test_model()

def load_model(name) -> tuple[model.PhiNet, model.HNet, dict]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")       
    package = torch.load(file_path)

    # recreate the model factory instance that generated the trained model
    model_factory_instance = model.ModelFactory(
        package["model_factory_init_args"]["num_of_conditions"],
        package["model_factory_init_args"]["dim_of_input"],
        package["model_factory_init_args"]["input_mean"],
        package["model_factory_init_args"]["input_scale"],
        package["model_factory_init_args"]["label_mean"],
        package["model_factory_init_args"]["label_scale"],
        customized_config=package["model_factory_init_args"]["customized_config"]
    )
    phi, h = model_factory_instance.generate_nets()
    phi.load_state_dict(package["phi"])
    phi.eval()  # set to eval mode
    h.load_state_dict(package["h"])
    h.eval()
    print("phi net input output fields:")
    for key, value in package["phi_net_io_fields"].items():
        print(key, value)
    return phi, h