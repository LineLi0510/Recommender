from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Model, layers

from domain.entities.train_data import TrainData
from domain.models.abstract_model import AbstractRecommenderBase, ModelTrainer
from domain.models.model_technologies.tensorflow.tf_utils import create_train_test_split, get_algo_params, \
    create_data_set


class TensorFlowRecommenderModel(AbstractRecommenderBase, ModelTrainer):
    def __init__(self, algo, algo_params, model_path: str, data_set: TrainData) -> None:
        super().__init__(model_path=model_path)
        self._algo = algo
        self._algo_params = algo_params
        self._model = self._init_model(data_set=data_set)
        self._train_data = None
        self._test_data = None
        self._tf_data_set = None

    @classmethod
    def create_model_from_dataset(
            cls,
            data_set: TrainData,
            algo,
            algo_params,
            model_path: str
    ) -> ModelTrainer:
        model = cls(algo=algo, algo_params=algo_params, model_path=model_path, data_set=data_set)
        model.run_model_training(data_set)

        return model

    def _init_model(self, data_set: TrainData) -> Model:
        self._algo_params = get_algo_params(algo_params=self._algo_params, data_set=data_set)

        return self._algo(algo_params=self._algo_params)

    def _fit_model(self) -> None:
        self._model.compile(
            optimizer=self._algo_params["optimizer"],
            loss=self._algo_params["loss"],
            metrics=self._algo_params["metrics"]
        )

        self._model.fit(
            x=self._train_data,
            validation_data=self._test_data,
            epochs=self._algo_params['epochs']
        )

    def dump_model(self) -> None:
        pass

    def _load_model(self, model_path: str) -> None:
        pass

    def calculate_model_metrics(self) -> Dict[str, Any]:
        predictions = self._model.predict([self._tf_data_set.user_tensor, self._tf_data_set.movie_tensor])
        labels_np = self._tf_data_set.rating_tensor.numpy()

        rmse = np.sqrt(mean_squared_error(y_true=labels_np, y_pred=np.squeeze(predictions)))
        loss, mae = self._model.evaluate([self._tf_data_set.user_tensor, self._tf_data_set.movie_tensor],
                                   self._tf_data_set.rating_tensor,
                                   verbose=0)
        model_metrics = {
            'loss': loss,
            'mae': mae,
            'rmse': rmse
        }

        return model_metrics

    def predict(self, user: int, item: int) -> float:
        pass

    def recommend(self, user: int, n: int) -> float:
        pass

    def run_model_training(self, data_set: TrainData) -> None:
        self._algo_params = get_algo_params(data_set=data_set, algo_params=self._algo_params)
        self._tf_data_set = create_data_set(data_set=data_set)
        self._train_data, self._test_data = create_train_test_split(data_set=self._tf_data_set)
        self._fit_model()
