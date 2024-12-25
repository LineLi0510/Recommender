from typing import Dict, Any

from surprise.model_selection import cross_validate

from domain.models.abstract_cross_validator import AbstractCrossValidator
from domain.models.model_technologies.surprise.surprise_utils import create_dataset
from src.domain.entities.train_data import TrainData


class SurpriseCrossValidator(AbstractCrossValidator):
    def __init__(self, algo, algo_params, cross_validation_setup: Dict[str, Any]) -> None:
        super().__init__(algo=algo, algo_params=algo_params, cross_validation_setup=cross_validation_setup)

    def cross_validate_model(self, data_set: TrainData) -> Dict[str, Any]:
        """
        Run cross validation on the model

        :param data_set: Dataset to run the cross validation on (TrainData)

        :return: A dictionary containing the cross validation
        """
        data_set = create_dataset(data_set=data_set)
        results = cross_validate(
            algo=self._algo,
            data=data_set,
            measures=self._cross_validation_setup['measures'],
            cv=self._cross_validation_setup['cv'],
            verbose=True
        )

        return results
