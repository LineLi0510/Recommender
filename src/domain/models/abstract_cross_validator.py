from abc import ABC, abstractmethod
from typing import Dict, Any

from src.domain.entities.train_data import TrainData


class AbstractCrossValidator(ABC):
    def __init__(
            self,
            algo,
            algo_params: Dict[str, Any],
            cross_validation_params: Dict[str, Any]
    ) -> None:
        self._algo = algo(**algo_params)
        self._cross_validation_setup = cross_validation_params

    @abstractmethod
    def cross_validate_model(self, train_data: TrainData) -> Dict[str, Any]:
        pass