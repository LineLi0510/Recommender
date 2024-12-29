from datetime import datetime
from typing import Type

from domain.entities.model_setup import ModelSetup
from domain.entities.train_data import TrainData
from domain.runner.runner_commands.runner_command import RunnerCommand
from domain.utils.ml_flow_tracker import MlFlowTracker


class TrainingRunnerCommand(RunnerCommand):
    runner_name = 'training'
    def __init__(
            self,
            model_setup: ModelSetup,
            tracker: Type[MlFlowTracker] | None = None
    ) -> None:
        self._runner_tracker = tracker
        self._model_setup = model_setup

    def execute(self, data_set: TrainData) -> None:
        try:
            if self._runner_tracker is not None:
                tracker = self._runner_tracker(experiment_name=f'{self.runner_name}/{self._model_setup.model_name}')
                tracker.start_run(run_name=f'{self._model_setup.model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}')
                model = self._model_setup.model_class.create_model_from_dataset(
                    data_set=data_set,
                    algo=self._model_setup.algo,
                    algo_params=self._model_setup.algo_params,
                    model_path=self._model_setup.model_path)
                model_kpis = model.calculate_model_metrics()
                tracker.log_params(self._model_setup.algo_params)
                tracker.log_metrics(model_kpis)
                tracker.end_run()
            else:
                model = self._model_setup.model_class.create_model_from_dataset(
                    data_set=data_set,
                    algo=self._model_setup.algo,
                    algo_params=self._model_setup.algo_params,
                    model_path=self._model_setup.model_path)
                model_kpis = model.calculate_model_metrics()
        except Exception as e:
            print('An error occurred while running training run', e)