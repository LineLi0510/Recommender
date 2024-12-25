import datetime
from functools import partial
from typing import Dict, Any

import optuna
from surprise import accuracy
from surprise.model_selection import cross_validate

from domain.models.model_technologies.surprise.surprise_utils import create_dataset, create_train_test_split
from domain.models.hyperparameter_tuning_technologies.optuna_hyperparameter_tuner import OptunaHyperparameterTuner
from src.domain.entities.train_data import TrainData


class SurpriseModelEvaluator:
    def __init__(self, model, model_setup) -> None:
        super().__init__(model=model, model_setup=model_setup)
        self.optimizer = OptunaHyperparameterTuner()

    def cross_validate_model(self, data_set: TrainData, cross_validation_setup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run cross validation on the model

        :param data_set: Dataset to run the cross validation on (TrainData)
        :param cross_validation_setup: Cross validation setup dictionary
        :return: A dictionary containing the cross validation
        """
        data_set = create_dataset(data_set=data_set)
        results = cross_validate(
            algo=self._model,
            data=data_set,
            measures=cross_validation_setup['measures'],
            cv=cross_validation_setup['cv'],
            verbose=True
        )

        return results

    def hyperparameter_objective(
            self,
            trial: optuna.trial.Trial,
            data_set: TrainData,
            hyperprameter_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        opt_data_set = create_dataset(data_set=data_set)
        train_set, test_set = create_train_test_split(opt_data_set)

        hyperparams = {
            param_name: self.optimizer.parse_hyper_parameters(param_name=param_name, param_values=param, trial=trial)
            for param_name, param in hyperprameter_config.items()
        }

        algo_class = self._model_setup.rec_model_config.algo
        algo_params = {**self._model_setup.rec_model_config.algo_params, **hyperparams}
        algo = algo_class(**algo_params)

        algo.fit(train_set)

        predictions = algo.test(test_set)
        rmse = accuracy.rmse(predictions, verbose=False)

        return rmse

    def run_hyperparameter_tuning(self, data_set: TrainData) -> None:
        hyperprameter_config = self._model_setup.hyperparameter_tuning_config
        if hyperprameter_config is not None:
            objective = partial(
                self.hyperparameter_objective,
                data_set=data_set,
                hyperprameter_config=hyperprameter_config
            )
            self.optimizer.run_optimization(
                objective=objective,
                study_name=self._model_setup.model_name + datetime.datetime.now().strftime("")
            )

