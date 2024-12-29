from datetime import datetime
from typing import Type

from domain.entities.model_setup import ModelSetup
from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand
from domain.utils.ml_flow_tracker import MlFlowTracker


class CrossValidationRunnerCommand(RunnerCommand):
    runner_name: str = 'cross_validation'

    def __init__(self, model_setup: ModelSetup, tracker: Type[MlFlowTracker]) -> None:
        self._model_setup = model_setup
        self._tracker = tracker

    def execute(self, data_set: TrainData) -> None:
        try:
            tracker = self._tracker(experiment_name=f'{self.runner_name}/{self._model_setup.model_name}')
            tracker.start_run(run_name=f'{self._model_setup.model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}')
            cross_validator = self._model_setup.cross_validator(
                algo=self._model_setup.algo,
                algo_params=self._model_setup.algo_params,
                cross_validation_params=self._model_setup.cross_validation_params
            )
            model_kpis = cross_validator.cross_validate_model(
                data_set=data_set,
            )
            tracker.log_cv_metrics(model_kpis)
            tracker.end_run()
        except Exception as e:
            print('An error occurred during cross validation: {}'.format(e))