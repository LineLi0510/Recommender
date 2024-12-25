from abc import ABC, abstractmethod
from typing import Dict, List

import mlflow
from mlflow import ActiveRun

from constants import mlf_tracking_uri


class AbstractTracker(ABC):
    @staticmethod
    @abstractmethod
    def log_metrics(metrics: Dict[str, float]):
        pass

    @staticmethod
    @abstractmethod
    def log_params(params: Dict[str, float]):
        pass


class MlFlowTracker(AbstractTracker):
    def __init__(self, experiment_name: str) -> None:
        mlflow.set_tracking_uri(mlf_tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.artifact_uri = 'artifacts'

    @staticmethod
    def log_metrics(metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    @staticmethod
    def log_cv_metrics(metrics: Dict[str, List[float]]) -> None:
        for kpi, runs in metrics.items():
            for value in runs:
                mlflow.log_metric(kpi, value)

    @staticmethod
    def log_params(params: Dict[str, float]) -> None:
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_artifacts(self, model_path: str) -> None:
        mlflow.log_artifact(local_path=model_path, artifact_path=self.artifact_uri)

    @staticmethod
    def start_run(run_name: str) -> ActiveRun:
        if mlflow.active_run() is None:
            return mlflow.start_run(run_name=run_name)
        else:
            return mlflow.start_run()

    @staticmethod
    def end_run() -> None:
        if mlflow.active_run() is not None:
            mlflow.end_run()
