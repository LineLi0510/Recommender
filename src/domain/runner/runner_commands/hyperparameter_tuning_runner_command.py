from domain.entities.model_setup import AbstractModelSetup
from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand
from domain.utils.ml_flow_tracker import MlFlowTracker


class HyperparameterTuningRunnerCommand(RunnerCommand):
    def __init__(self, model_config: AbstractModelSetup, tracker: MlFlowTracker | None = None) -> None:
        super().__init__(tracker=tracker, model_config=model_config)

    def execute(self, data_set: TrainData) -> None:
        try:
            model_hyperparameter_tuner = self._model_config.hyperparameter_tuner()
            model_hyperparameter_tuner.run_hyperparameter_tuning(
                model_setup=self._model_config,
                data_set=data_set,
                objective_method=self._model_config.hyperparameter_objective(model_setup=self._model_config).objective
            )
        except Exception as e:
            print('error during hyperparameter tuning', e)