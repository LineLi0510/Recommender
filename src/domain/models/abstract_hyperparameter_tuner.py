from abc import ABC


class AbstractHyperparameterTuner(ABC):
    def run_hyperparameter_tuning(self, model_setup, data_set, objective_method):
        pass