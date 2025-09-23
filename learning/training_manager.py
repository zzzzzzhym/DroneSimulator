import os
import torch

import model
import data_factory
import trainer
import validator

class DataFactoryArtifacts:
    def __init__(self) -> None:
        # DIAML specific
        self.loaderset_phi = None 
        self.loaderset_a = None
        self.num_of_conditions = None 
        # simple specific
        self.datasets = None
        self.loaderset = None
        # common
        self.dim_of_input = None 
        self.dim_of_label = None 
        self.input_mean_vector = None
        self.input_scale_vector = None
        self.label_mean_vector = None
        self.label_scale_vector = None
        self.input_label_map = None  # a dict mapping input and label names to column indices in data files

class ModelFactoryArtifacts:
    def __init__(self) -> None:
        # DIAML specific
        self.phi_net = None
        self.h_net = None
        # simple specific
        self.simple_net = None        
        # common
        self.config = None


class ValidatorArtifacts:
    def __init__(self) -> None:
        self.in_run_validate = None  # callable function for validation during training


class Adapter:
    """Encapsulates a factory instance and provides a uniform interface to get the package for training manager"""
    def __init__(self) -> None:
        self.implementation = None

    def set_up(self, specs):
        raise NotImplementedError("Adapter subclasses must implement set_up_factory method")

    def generate_artifacts(self) -> object:
        raise NotImplementedError("Adapter subclasses must implement generate_artifacts method")

class DiamlDataFactoryAdapter(Adapter):
    def set_up(self, sample_data_menu: list, input_label_map_file: str) -> None:
        """The sample data menu is a subset of training data to inspect number of conditions and do normalization"""
        print("Setting up data factory...")
        self.implementation = data_factory.DiamlDataFactory(
            input_label_map_file
        )
        self.implementation.set_num_of_conditions(sample_data_menu)
        self.implementation.make_normalization_params(sample_data_menu)

    def generate_artifacts(self, data_menu: list, can_inspect_data: bool=False) -> DataFactoryArtifacts:
        artifacts = DataFactoryArtifacts()
        artifacts.datasets = self.implementation.prepare_datasets(data_menu, can_inspect_data)
        artifacts.loaderset_phi, artifacts.loaderset_a = self.implementation.prepare_loadersets(artifacts.datasets)
        artifacts.dim_of_input = len(self.implementation.input_headers)
        artifacts.dim_of_label = len(self.implementation.label_headers)
        artifacts.num_of_conditions = self.implementation.num_of_conditions
        artifacts.input_mean_vector = self.implementation.input_mean_vector
        artifacts.input_scale_vector = self.implementation.input_scale_vector
        artifacts.label_mean_vector = self.implementation.label_mean_vector
        artifacts.label_scale_vector = self.implementation.label_scale_vector
        artifacts.input_label_map = self.implementation.input_label_map
        return artifacts

class DiamlModelFactoryAdapter(Adapter):
    def set_up(self, artifacts: DataFactoryArtifacts) -> None:
        print("Setting up model factory...")
        self.implementation = model.DiamlModelFactory(
            artifacts.num_of_conditions,
            artifacts.dim_of_input,
            artifacts.input_mean_vector,
            artifacts.input_scale_vector,
            artifacts.label_mean_vector,
            artifacts.label_scale_vector,
        )

    def generate_artifacts(self) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.phi_net, artifacts.h_net = self.implementation.generate_nets()
        artifacts.config = self.implementation.generate_self_config()
        return artifacts

class DiamlTrainerAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts, artifacts_validator: ValidatorArtifacts) -> None:
        print("Setting up trainer...")
        self.implementation = trainer.Trainer(
            artifacts_model.phi_net,
            artifacts_model.h_net,
            artifacts_data.loaderset_phi,
            artifacts_data.loaderset_a,
            artifacts_data.dim_of_label,
            artifacts_validator.in_run_validate
        )

    def train(self):
        self.implementation.train_model()
        self.implementation.plot_loss()

    def generate_results(self):
        """Trigger to train the model"""
        self.implementation.plot_loss()
        self.implementation.plot_tsne_of_a_trace()

class DiamlModelSaver:
    def set_up(self, artifacts_model: ModelFactoryArtifacts, artifacts_data: DataFactoryArtifacts) -> None:
        self.phi_net = artifacts_model.phi_net
        self.h_net = artifacts_model.h_net
        self.model_factory_config = artifacts_model.config
        self.input_label_map = artifacts_data.input_label_map

    def save_model(self, name) -> None:
        model.save_diaml_model(name, self.phi_net, self.h_net, self.model_factory_config, self.input_label_map)

class DiamlModelLoader:
    def generate_artifacts(self, name) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.phi_net, artifacts.h_net = model.load_diaml_model(name)
        return artifacts
        
class DiamlValidatorAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts) -> None:
        print("Setting up validator...")
        self.validator_instance = validator.DiamlEvaluator()
        self.validator_instance.load_model(artifacts_model.phi_net, artifacts_model.h_net)
        self.validator_instance.load_dataset(artifacts_data.datasets)

    def generate_artifacts(self) -> ValidatorArtifacts:
        artifacts = ValidatorArtifacts()
        artifacts.in_run_validate = self.validator_instance.callback_validation
        return artifacts
    
    def test_model(self) -> None:
        self.validator_instance.test_model()

class SimpleDataFactoryAdapter(Adapter):
    def set_up(self, sample_data_menu: list, input_label_map_file: str) -> None:
        """The sample data menu is a subset of training data to inspect number of conditions and do normalization"""
        print("Setting up data factory...")
        self.implementation = data_factory.SimpleDataFactory(
            input_label_map_file
        )
        self.implementation.make_normalization_params(sample_data_menu)

    def generate_artifacts(self, data_menu: list, can_inspect_data: bool=False) -> DataFactoryArtifacts:
        artifacts = DataFactoryArtifacts()
        artifacts.datasets = self.implementation.prepare_datasets(data_menu, can_inspect_data)
        artifacts.loaderset = self.implementation.prepare_loaderset(artifacts.datasets)
        artifacts.dim_of_input = len(self.implementation.input_headers)
        artifacts.dim_of_label = len(self.implementation.label_headers)
        artifacts.input_mean_vector = self.implementation.input_mean_vector
        artifacts.input_scale_vector = self.implementation.input_scale_vector
        artifacts.label_mean_vector = self.implementation.label_mean_vector
        artifacts.label_scale_vector = self.implementation.label_scale_vector
        artifacts.input_label_map = self.implementation.input_label_map
        return artifacts


class SimpleModelFactoryAdapter(Adapter):
    def set_up(self, artifacts: DataFactoryArtifacts) -> None:
        print("Setting up model factory...")
        self.implementation = model.SimpleNetFactory(
            artifacts.dim_of_input,
            artifacts.dim_of_label,
            artifacts.input_mean_vector,
            artifacts.input_scale_vector,
            artifacts.label_mean_vector,
            artifacts.label_scale_vector,
        )

    def generate_artifacts(self) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.simple_net = self.implementation.generate_nets()
        artifacts.config = self.implementation.generate_self_config()
        return artifacts

class SimpleTrainerAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts, artifacts_validator: ValidatorArtifacts) -> None:
        print("Setting up trainer...")
        self.implementation = trainer.SimpleTrainer(
            artifacts_model.simple_net,
            artifacts_data.loaderset,
            artifacts_data.dim_of_label,
            artifacts_validator.in_run_validate
        )

    def train(self):
        self.implementation.train_model()
        self.implementation.plot_loss()

    def generate_results(self):
        """Trigger to train the model"""
        self.implementation.plot_loss()

class SimpleModelSaver:
    def set_up(self, artifacts_model: ModelFactoryArtifacts, artifacts_data: DataFactoryArtifacts) -> None:
        self.simple_net = artifacts_model.simple_net
        self.model_factory_config = artifacts_model.config
        self.input_label_map = artifacts_data.input_label_map

    def save_model(self, name) -> None:
        model.save_simple_model(name, self.simple_net, self.model_factory_config, self.input_label_map)

class SimpleModelLoader:
    def generate_artifacts(self, name) -> ModelFactoryArtifacts:
        artifacts = ModelFactoryArtifacts()
        artifacts.simple_net = model.load_simple_model(name)
        return artifacts

        
class SimpleValidatorAdapter(Adapter):
    def set_up(self, artifacts_data: DataFactoryArtifacts, artifacts_model: ModelFactoryArtifacts) -> None:
        print("Setting up validator...")
        self.validator_instance = validator.SimpleEvaluator()
        self.validator_instance.load_model(artifacts_model.simple_net)
        self.validator_instance.load_dataset(artifacts_data.datasets)

    def generate_artifacts(self) -> ValidatorArtifacts:
        artifacts = ValidatorArtifacts()
        artifacts.in_run_validate = self.validator_instance.callback_validation
        return artifacts
    
    def test_model(self) -> None:
        self.validator_instance.test_model()




class TrainingPipeline:
    def __init__(self, data_adapter: Adapter, model_adapter: Adapter, trainer_adapter: Adapter, validator_adapter: Adapter, model_saver) -> None:
        self.data_adapter = data_adapter
        self.model_adapter = model_adapter
        self.trainer_adapter = trainer_adapter
        self.validator_adapter = validator_adapter
        self.model_saver = model_saver

    def set_up(self, training_data_menu: list, validation_data_menu: list, input_label_map_file: str, can_inspect_data=False) -> None:
        self.data_adapter.set_up(training_data_menu, input_label_map_file)
        training_data_artifacts = self.data_adapter.generate_artifacts(training_data_menu, can_inspect_data)
        validation_data_artifacts = self.data_adapter.generate_artifacts(validation_data_menu, can_inspect_data)
        self.model_adapter.set_up(training_data_artifacts)
        model_artifacts = self.model_adapter.generate_artifacts()
        self.validator_adapter.set_up(validation_data_artifacts, model_artifacts)
        validator_artifacts = self.validator_adapter.generate_artifacts()
        self.trainer_adapter.set_up(training_data_artifacts, model_artifacts, validator_artifacts)
        self.model_saver.set_up(model_artifacts, training_data_artifacts)

    def train(self) -> None:
        self.trainer_adapter.train()

    def show_result_only(self) -> None:
        self.trainer_adapter.generate_results()

    def save_model(self, name):
        self.model_saver.save_model(name)


class TestPipeline:
    def __init__(self, data_adapter: Adapter, validator_adapter: Adapter, model_loader: DiamlModelLoader) -> None:
        self.data_adapter = data_adapter
        self.validator_adapter = validator_adapter
        self.model_loader = model_loader

    def set_up(self, data_menu: list, input_label_map_file: str, model_name: str) -> None:
        self.data_adapter.set_up(data_menu, input_label_map_file)
        data_artifacts = self.data_adapter.generate_artifacts(data_menu)
        model_artifacts = self.model_loader.generate_artifacts(model_name)
        self.validator_adapter.set_up(data_artifacts, model_artifacts)

    def test(self) -> None:
        self.validator_adapter.test_model()

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
               ) -> None:
        self.set_up_data_factory(data_menu, input_label_map_file)
        self.set_up_tester()

    def set_up_data_factory(self, data_menu, input_label_map_file: str) -> None:
        self.data_factory_instance = data_factory.DiamlDataFactory(
            input_label_map_file
        )
        self.data_factory_instance.make_normalization_params(data_menu)
        self.dim_of_input = len(self.data_factory_instance.input_headers)
        self.dim_of_label = len(self.data_factory_instance.label_headers)
        self.num_of_conditions = self.data_factory_instance.num_of_conditions # assume each data file has a unique condition

    def set_up_tester(self) -> None:
        self.tester_instance = validator.DiamlEvaluator()

    def test(self, phi_net: model.MultilayerNet, h_net: model.MultilayerNet, data_menu_validation: list) -> None:
        datasets = self.data_factory_instance.prepare_datasets(data_menu_validation, False)        
        self.tester_instance.load_model(phi_net, h_net)
        self.tester_instance.load_dataset(datasets)
        self.tester_instance.test_model()


class PipelineFactory:
    """An abstract factory selects and instruct the detailed factories to create parts"""
    def __init__(self, is_diaml) -> None:
        self.config = self.load_config(is_diaml)

    def load_config(self, is_diaml: bool) -> None:
        """Simple config for now, can load from a file later"""
        config = {}
        config["is_diaml"] = is_diaml
        return config

    def make_training_pipeline(self) -> TrainingPipeline:
        """API for users to get the training pipeline, the main job is to select the right adapters"""
        if self.config["is_diaml"]:
            return TrainingPipeline(
                DiamlDataFactoryAdapter(),
                DiamlModelFactoryAdapter(),
                DiamlTrainerAdapter(),
                DiamlValidatorAdapter(),
                DiamlModelSaver()
            )
        else:
            return TrainingPipeline(
                SimpleDataFactoryAdapter(),
                SimpleModelFactoryAdapter(),
                SimpleTrainerAdapter(),
                SimpleValidatorAdapter(),
                SimpleModelSaver()
            )

    def make_test_pipeline(self) -> TestPipeline:
        """API for users to get the testing pipeline, the main job is to select the right adapters"""
        if self.config["is_diaml"]:
            return TestPipeline(
                DiamlDataFactoryAdapter(),
                DiamlValidatorAdapter(),
                DiamlModelLoader()
            )
        else:
            return TestPipeline(
                SimpleDataFactoryAdapter(),
                SimpleValidatorAdapter(),
                SimpleModelLoader()
            )
