from enum import Enum
import pipeline_adapters

class ModelArchitecture(Enum):
    DIAML = 0
    SIMPLE_NET = 1
    ROTOR_NET = 2

class TrainingPipeline:
    def __init__(
        self,
        data_adapter: pipeline_adapters.Adapter,
        model_adapter: pipeline_adapters.Adapter,
        trainer_adapter: pipeline_adapters.Adapter,
        validator_adapter: pipeline_adapters.Adapter,
        model_saver
    ) -> None:
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
    def __init__(
        self,
        data_adapter: pipeline_adapters.Adapter,
        validator_adapter: pipeline_adapters.Adapter,
        model_loader: pipeline_adapters.DiamlModelLoader
    ) -> None:
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
        

class PipelineFactory:
    """An abstract factory selects and instruct the detailed factories to create parts"""
    PIPELINE_BUILDERS = {
        ModelArchitecture.DIAML: {
            "train": lambda: TrainingPipeline(
                pipeline_adapters.DiamlDataFactoryAdapter(),
                pipeline_adapters.DiamlModelFactoryAdapter(),
                pipeline_adapters.DiamlTrainerAdapter(),
                pipeline_adapters.DiamlValidatorAdapter(),
                pipeline_adapters.DiamlModelSaver(),
            ),
            "test": lambda: TestPipeline(
                pipeline_adapters.DiamlDataFactoryAdapter(),
                pipeline_adapters.DiamlValidatorAdapter(),
                pipeline_adapters.DiamlModelLoader(),
            ),
        },

        ModelArchitecture.SIMPLE_NET: {
            "train": lambda: TrainingPipeline(
                pipeline_adapters.SimpleDataFactoryAdapter(),
                pipeline_adapters.SimpleModelFactoryAdapter(),
                pipeline_adapters.SimpleTrainerAdapter(),
                pipeline_adapters.SimpleValidatorAdapter(),
                pipeline_adapters.SimpleModelSaver(),
            ),
            "test": lambda: TestPipeline(
                pipeline_adapters.SimpleDataFactoryAdapter(),
                pipeline_adapters.SimpleValidatorAdapter(),
                pipeline_adapters.SimpleModelLoader(),
            ),
        },

        ModelArchitecture.ROTOR_NET: {
            "train": lambda: TrainingPipeline(
                pipeline_adapters.RotorNetDataFactoryAdapter(),
                pipeline_adapters.RotorNetModelFactoryAdapter(),
                pipeline_adapters.RotorNetTrainerAdapter(),
                pipeline_adapters.RotorNetValidatorAdapter(),
                pipeline_adapters.RotorNetModelSaver(),
            ),
            "test": lambda: TestPipeline(
                pipeline_adapters.RotorNetDataFactoryAdapter(),
                pipeline_adapters.RotorNetValidatorAdapter(),
                pipeline_adapters.RotorNetModelLoader(),
            ),
        },
    }

    def __init__(self, model_type: ModelArchitecture) -> None:
        self.config = self.load_config(model_type)

    def load_config(self, model_type: ModelArchitecture) -> None:
        """Simple config for now, can load from a file later"""
        config = {}
        config["model_type"] = model_type
        return config

    def make_training_pipeline(self) -> TrainingPipeline:
        """API for users to get the training pipeline, the main job is to select the right adapters"""
        return self.PIPELINE_BUILDERS[self.config["model_type"]]["train"]()

    def make_test_pipeline(self) -> TestPipeline:
        """API for users to get the testing pipeline, the main job is to select the right adapters"""
        return self.PIPELINE_BUILDERS[self.config["model_type"]]["test"]()


import model
import data_factory
import trainer
import validator

class TestManager:  # deprecated
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