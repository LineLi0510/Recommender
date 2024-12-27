from abc import ABC, abstractmethod


class AbstractHyperparameterTuner(ABC):
    @abstractmethod
    def run_hyperparameter_tuning(self, model_setup, data_set, objective_method):
        pass