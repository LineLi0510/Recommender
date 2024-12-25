from typing import Dict, Any, Optional

from surprise import Dataset, dump, accuracy

from domain.entities.train_data import TrainData
from domain.models.abstract_model import AbstractRecommenderBase, ModelTrainer, ModelInferer
from domain.models.model_technologies.surprise.surprise_utils import create_dataset, create_train_test_split


class SurpriseRecommenderModel(AbstractRecommenderBase, ModelTrainer, ModelInferer):
    def __init__(self, algo, algo_params, model_path: str):
        super().__init__(model_path=model_path)
        self._algo = algo
        self._algo_params = algo_params
        self._model = algo(**algo_params)
        self._train_data: Optional[Dataset] = None
        self._test_data: Optional[Dataset] = None

    @classmethod
    def create_model_from_dataset(
            cls,
            data_set: TrainData,
            algo,
            algo_params,
            model_path: str
    ) -> ModelTrainer:
        model = cls(algo=algo, algo_params=algo_params, model_path=model_path)
        model.run_model_training(data_set)

        return model

    @classmethod
    def load_from_file(cls, model_path: str) -> ModelInferer:
        model = cls.__new__(cls)
        model._load_model(model_path)

        return model

    def _create_train_test_split(self, data_set: Dataset) -> None:
        self._train_data, self._test_data = create_train_test_split(data_set=data_set)

    def _fit_model(self) -> None:
        self._model.fit(self._train_data)

    def dump_model(self) -> None:
        dump.dump(file_name=self.model_path, algo=self._model)

    def _load_model(self, model_path: str) -> None:
        self._model = dump.load(file_name=model_path)

    def calculate_model_metrics(self) -> Dict[str, Any]:
        if self._test_data is None:
            raise ValueError("No test data provided. Create model from dataset first. ")

        predictions = self._model.test(self._test_data)
        model_metrics = {
            "RMSE": accuracy.rmse(predictions, verbose=False),
            "MAE": accuracy.mae(predictions, verbose=False)
        }

        return model_metrics

    def predict(self, user: int, item: int) -> float:
        pass

    def recommend(self, user: int, n: int) -> float:
        pass

    def run_model_training(self, data_set: TrainData) -> None:
        data_set = create_dataset(data_set=data_set)
        self._create_train_test_split(data_set=data_set)
        self._fit_model()
