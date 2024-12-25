from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any
from src.domain.entities.train_data import TrainData


class ModelTrainer(Protocol):
    def run_model_training(self, train_data: TrainData) -> None:
        pass

    def calculate_model_metrics(self) -> Dict[str, Any]:
        pass


class ModelInferer(Protocol):
    def predict(self, user: int, item: int) -> float:
        pass

    def recommend(self, user: int, n: int) -> float:
        pass

class AbstractRecommenderBase(ABC):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    @abstractmethod
    def run_model_training(self, train_data: TrainData) -> None:
        pass

    @abstractmethod
    def calculate_model_metrics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, user: int, item: int) -> float:
        pass

    @abstractmethod
    def recommend(self, user: int, n: int) -> float:
        pass
