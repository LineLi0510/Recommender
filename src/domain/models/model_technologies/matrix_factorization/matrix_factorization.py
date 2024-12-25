from sklearn.decomposition import TruncatedSVD

from domain.entities.train_data import TrainData
from src.domain.models.abstract_model import AbstractRecommenderBase
import pandas as pd
import numpy as np

from src.domain.utils.ml_flow_tracker import AbstractTracker


class MatrixFactorizationRecommender(AbstractRecommenderBase):
    def __init__(
            self,
            user_col: str,
            item_col: str,
            rating_col: str,
            n_components: int = 10,
            tracker: AbstractTracker = None
    ) -> None:
        super().__init__(user_col, item_col, rating_col, tracker)
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=self.n_components)

    def run_model_training(self, data: TrainData) -> None:
        self.user_factors = self.model.fit_transform(data.sparse_matrix)
        self.item_factors = self.model.components_
        self.trained = True
        self.tracker

    def predict(self, user: int, item: int) -> float:
        if not self.trained:
            raise ValueError("Das Modell muss zuerst trainiert werden.")
        return np.dot(self.user_factors[user], self.item_factors[:, item])

    def recommend(self, user: int, n: int = 10) -> list:
        if not self.trained:
            raise ValueError("Das Modell muss zuerst trainiert werden.")
        scores = np.dot(self.user_factors[user], self.item_factors)
        return np.argsort(-scores)[:n]