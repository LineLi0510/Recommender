from domain.model_training.data_loader.raw_data_loader import RawDataLoader
from domain.model_training.data_processing.train_data_creator import TrainDataCreator
from domain.model_training.train_config import DataProcessingConfig
from domain.runner.runner_commands.cross_validation_runner_command import CrossValidationRunnerCommand
from domain.runner.runner_commands.hyperparameter_tuning_runner_command import HyperparameterTuningRunnerCommand
from domain.runner.runner_commands.training_runner_command import TrainingRunnerCommand
from domain.runner.runner_invoker import RunnerInvoker
from model_training_pipeline import models_to_train
from persistence.database.database_setup import engine
from persistence.db_service import DbService
from persistence.entities.base import Base
from src.domain.utils.ml_flow_tracker import MlFlowTracker


Base.metadata.create_all(engine)
chunksize = 100000

TRACKER = MlFlowTracker


def run_training():
    db_service = DbService(engine)
    raw_data = RawDataLoader(db_service=db_service).process()
    data_set = TrainDataCreator(train_config=DataProcessingConfig()).process(raw_data)

    for model_config in models_to_train:
        runner_commands = [
            TrainingRunnerCommand(
                tracker=TRACKER(experiment_name=f'training/{model_config.model_name}'),
                model_config=model_config
            ),
            CrossValidationRunnerCommand(
                tracker=TRACKER(experiment_name=f'cross_validation/{model_config.model_name}'),
                model_config=model_config
            ),
            HyperparameterTuningRunnerCommand(
                tracker=None,
                model_config=model_config
            )
        ]
        RunnerInvoker().run_commands(runner_commands=runner_commands, data_set=data_set)


if __name__ == '__main__':
    run_training()
