import pandas as pd
from surprise import Dataset, Reader
from typing import List, Dict, Union
from abc import ABC, abstractmethod
from domain.entities.ratings import Ratings
from surprise.model_selection import cross_validate
from surprise import dump
from domain.model_training.model.surprise.model_training_pipeline import AlgoT
import mlflow


class AbstractModel(ABC):
    @staticmethod
    def _create_dataset(ratings: List[Ratings]) -> Dataset:
        data = pd.DataFrame([(rating.user_id, rating.movie_id, rating.rating) for rating in ratings])
        reader = Reader(rating_scale=(0.5, 5))
        dataset = Dataset.load_from_df(df=data, reader=reader)

        return dataset

    @abstractmethod
    def process(self, ratings: List[Ratings], algo: AlgoT) -> None:
        pass


class SurpriseModelCV(AbstractModel):
    @staticmethod
    def _cv_model(train_set, model_config: Dict[str, Union[str, AlgoT]]):
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("recommender_test")
        with mlflow.start_run(run_name='CV_' + model_config['model_name']):
            results = cross_validate(algo=model_config['algo'], data=train_set, measures=['RMSE', 'MAE'], cv=5, verbose=True)
            mlflow.log_metric("test_rmse", results['test_rmse'][0])
            mlflow.log_metric("test_mae", results['test_mae'][0])

    def process(self, ratings: List[Ratings], model_configs: List[Dict[str, Union[str, AlgoT]]]) -> None:
        train_data = self._create_dataset(ratings=ratings)
        for model_config in model_configs:
            self._cv_model(train_set=train_data, model_config=model_config)



class SurpriseModel(AbstractModel):
    @staticmethod
    def _train_model(train_set, model_config: Dict[str, Union[str, AlgoT]]):
        train_set = train_set.build_full_trainset()
        with mlflow.start_run(run_name='Training_' + model_config['model_name']):
            algo = model_config['algo']
            algo.fit(train_set)

    @staticmethod
    def _dump_model(model_config: Dict[str, Union[str, AlgoT]]):
        dump.dump(file_name=model_config['model_saving_path'], algo=model_config['algo'])

    def process(self, ratings: List[Ratings], model_configs: List[Dict[str, Union[str, AlgoT]]]) -> None:
        train_data = self._create_dataset(ratings=ratings)
        for model_config in model_configs:
            self._train_model(train_set=train_data, model_config=model_config)
            self._dump_model(model_config=model_config)

