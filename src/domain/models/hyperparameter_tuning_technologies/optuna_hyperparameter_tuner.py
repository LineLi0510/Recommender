from datetime import datetime
from functools import partial
from typing import Dict, Any, Callable, Union

import optuna
from optuna.storages import RDBStorage
from optuna.visualization import plot_optimization_history, plot_param_importances

from domain.entities.train_data import TrainData
from domain.models.abstract_hyperparameter_tuner import AbstractHyperparameterTuner


class OptunaHyperparameterTuner(AbstractHyperparameterTuner):
    def __init__(self):
        self._storage = RDBStorage(url="mysql+mysqlconnector://user:password@optuna-db/optuna")
        self.hyperparameter_config = None

    def _check_study_existence(self, study_name: str) -> bool:
        """
        Checks if the study exists in the db

        :param study_name: name of the study
        :return: True if study exists, False otherwise
        """
        all_studies = optuna.study.get_all_study_summaries(storage=self._storage)
        study_exists = any(study.study_name == study_name for study in all_studies)

        return study_exists

    @staticmethod
    def _show_result(study: optuna.study.Study) -> None:
        """
        Print out the results of the study

        :param study: optuna study
        """
        print('Best trial:', study.best_trial)
        print('Min value:', study.best_value)

    @staticmethod
    def _visualize_result(study: optuna.study) -> None:
        """
        Visualize the results of the optimization study

        :param study: optuna.study.Study
        """
        plot_optimization_history(study).show()
        plot_param_importances(study).show()

    def _trial_handler(
            self,
            trial: optuna.trial.Trial,
            objective: Callable[[Dict[str, Any]], float]
    ) -> Callable[[Dict[str, Any]], float]:
        """
        Trial handler

        :param trial: trial object
        :param objective: objective function

        :return: objective function with parameter values
        """
        hyperparameters = {
            param_name: self._parse_hyper_parameters(param_name, param_values, trial)
            for param_name, param_values in self.hyperparameter_config.items()
        }
        return objective(hyperparameters)

    @staticmethod
    def _parse_hyper_parameters(
            param_name: str,
            param_values: Dict[str, Any],
            trial: optuna.trial.Trial
    ) -> Union[None, bool, int, float, str]:
        """
        Parse hyperparameter parameters from trial and return the corresponding

        :param param_name: name of the hyperparameter to tune
        :param param_values: the values of the hyperparameter
        :param trial: the current trial

        :return: the corresponding hyperparameter value
        """
        if param_values['type'] == 'int':
            return trial.suggest_int(name=param_name, low=param_values['low'], high=param_values['high'])
        elif param_values['type'] == 'loguniform':
            return trial.suggest_loguniform(param_name, param_values['low'], param_values['high'])
        elif param_values['type'] == 'categorical':
            return trial.suggest_categorical(param_name, param_values['values'])
        else:
            raise ValueError(f"Unsupported parameter type: {param_values['type']}")

    def _run_optimization(self, study_name: str, trial_handler: Callable[[optuna.trial.Trial], float]) -> None:
        """
        Run the optimization

        :param study_name: The name of the study to run the optimization
        :param trial_handler: The function to run the optimization
        """
        if not self._check_study_existence(study_name):
            study = optuna.create_study(study_name=study_name, storage=self._storage, direction="minimize")
        else:
            study = optuna.load_study(study_name=study_name, storage=self._storage)

        study.optimize(trial_handler, n_trials=10)
        self._show_result(study=study)

    def run_hyperparameter_tuning(
            self,
            model_setup,
            data_set: TrainData,
            objective_method: Callable
    ) -> None:
        """
        Run the optimization

        :param model_setup: The model setup
        :param data_set: The data set to be used for the optimization
        :param objective_method: The objective function to be used for the optimization
        """
        hyperparameter_config = model_setup.hyperparameter_tuning_config
        if hyperparameter_config is not None:
            self.hyperparameter_config = hyperparameter_config
            objective = partial(
                objective_method,
                data_set=data_set,
            )
            study_name = model_setup.model_name +'_' + datetime.now().strftime("%Y%m%d")
            self._run_optimization(
                study_name=study_name,
                trial_handler=lambda trial: self._trial_handler(trial, objective)
            )
