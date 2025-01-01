from domain.entities.model_setup import ModelSetup
from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand


class HyperparameterTuningRunnerCommand(RunnerCommand):
    def __init__(self, model_setup: ModelSetup) -> None:
        self._model_setup = model_setup

    def execute(self, data_set: TrainData) -> None:
        try:
            if self._model_setup.hyperparameter_tuning_config:
                model_hyperparameter_tuner = self._model_setup.hyperparameter_tuner
                model_hyperparameter_tuner.run_hyperparameter_tuning(
                    model_setup=self._model_setup,
                    data_set=data_set,
                    objective_method=self._model_setup.hyperparameter_objective(model_setup=self._model_setup).objective
                )
        except Exception as e:
            print('error during hyperparameter tuning', e)