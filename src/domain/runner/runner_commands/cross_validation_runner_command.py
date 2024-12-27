from datetime import datetime

from domain.entities.model_setup import AbstractModelSetup
from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand
from domain.utils.ml_flow_tracker import MlFlowTracker


class CrossValidationRunnerCommand(RunnerCommand):
    def __init__(self, model_config: AbstractModelSetup, tracker: MlFlowTracker | None = None) -> None:
        super().__init__(tracker=tracker, model_config=model_config)

    def execute(self, data_set: TrainData) -> None:
        try:
            self._tracker.start_run(run_name=f'{self._model_config.model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}')
            cross_validator = self._model_config.cross_validator(
                algo=self._model_config.algo,
                algo_params=self._model_config.algo_params,
                cross_validation_setup=self._model_config.cross_validation_params
            )
            model_kpis = cross_validator.cross_validate_model(
                data_set=data_set,
            )
            self._tracker.log_cv_metrics(model_kpis)
            self._tracker.end_run()
        except Exception as e:
            print('An error occurred during cross validation: {}'.format(e))