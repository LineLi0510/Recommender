from domain.entities.model_setup import AbstractModelSetup
from domain.entities.train_data import TrainData
from domain.utils.ml_flow_tracker import MlFlowTracker


class RunnerCommand:
    def __init__(self, model_config: AbstractModelSetup, tracker: MlFlowTracker | None = None) -> None:
        self._tracker = tracker
        self._model_config = model_config

    @staticmethod
    def execute(self, data_set: TrainData) -> None:
        pass