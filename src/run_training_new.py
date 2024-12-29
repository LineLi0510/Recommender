from domain.model_training.data_loader.raw_data_loader import RawDataLoader
from domain.model_training.data_processing.train_data_creator import TrainDataCreator
from domain.model_training.train_config import DataProcessingConfig
from domain.runner.runner_commands.cross_validation_runner_command import CrossValidationRunnerCommand
from domain.runner.runner_commands.hyperparameter_tuning_runner_command import HyperparameterTuningRunnerCommand
from domain.runner.runner_commands.training_runner_command import TrainingRunnerCommand
from domain.runner.runner_invoker import RunnerInvoker
from domain.utils.ml_flow_tracker import MlFlowTracker
from model_training_pipeline_new import models_to_train
from persistence.database.database_setup import engine
from persistence.db_service import DbService
from persistence.entities.base import Base


Base.metadata.create_all(engine)
chunksize = 100000

train_models = True
run_cross_validation = True
run_hyperparametertuning = True

TRACKER = MlFlowTracker


def run_training():
    db_service = DbService(engine)
    raw_data = RawDataLoader(db_service=db_service).process()
    data_set = TrainDataCreator(train_config=DataProcessingConfig()).process(raw_data)

    for model_setup in models_to_train:
        runner_commands = []
        if train_models:
            runner_commands.append(TrainingRunnerCommand(tracker=TRACKER, model_setup=model_setup))
        if run_cross_validation:
            runner_commands.append(CrossValidationRunnerCommand(tracker=TRACKER, model_setup=model_setup))
        if run_hyperparametertuning:
            runner_commands.append(HyperparameterTuningRunnerCommand(model_setup=model_setup))

        RunnerInvoker(runner_commands=runner_commands).run_commands(data_set=data_set)


if __name__ == '__main__':
    run_training()
