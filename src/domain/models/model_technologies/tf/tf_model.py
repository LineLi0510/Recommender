from typing import Dict, Any

from domain.entities.train_data import TrainData
from domain.models.abstract_model import AbstractRecommenderBase


class TensorFlowRecommenderModel(AbstractRecommenderBase):
    def __init__(self, rec_model_config, model_path: str):
        super().__init__(rec_model_config=rec_model_config, model_path=model_path)

    def predict(self, user: int, item: int) -> float:
        pass

    def recommend(self, user: int, n: int) -> float:
        pass

    def run_model_training(self, data_set: TrainData) -> Dict[str, Any]:
        pass

    def cv_model(self, data_set: TrainData):
        pass
