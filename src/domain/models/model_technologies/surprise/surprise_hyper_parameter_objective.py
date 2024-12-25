from typing import Dict, Any

from surprise import accuracy

from domain.models.model_technologies.surprise.surprise_utils import create_dataset, create_train_test_split
from src.domain.entities.train_data import TrainData


class SurpriseHyperparameterObjective:
    def __init__(self, model_setup) -> None:
        self._model_setup = model_setup

    def objective(
            self,
            hyperparameters: Dict[str, Any],
            data_set: TrainData,
    ) -> float:
        opt_data_set = create_dataset(data_set=data_set)
        train_set, test_set = create_train_test_split(opt_data_set)

        algo_class = self._model_setup.algo
        algo_params = {**self._model_setup.algo_params, **hyperparameters}
        algo = algo_class(**algo_params)

        algo.fit(train_set)
        predictions = algo.test(test_set)
        rmse = accuracy.rmse(predictions, verbose=False)

        return rmse


