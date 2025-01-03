import pickle
from typing import Tuple, Dict

from scipy.sparse import csr_matrix

from domain.entities.ratings import RatingData
from domain.entities.train_data import TrainData
from domain.model_training.train_config import DataProcessingConfig


class TrainDataCreator:
    def __init__(self, train_config: DataProcessingConfig) -> None:
        self.train_config = train_config

    @staticmethod
    def _create_id_mapping(id_list) -> Tuple[Dict[int, int], Dict[int, int]]:
        id_to_feature_id = {i: object_id for i, object_id in enumerate(set(id_list))}
        feature_id_to_id = {object_id: i for i, object_id in enumerate(set(id_list))}

        return feature_id_to_id, id_to_feature_id

    @staticmethod
    def _save_data(train_data: TrainData) -> None:
        with open('/data/train_data/ratings_train.pkl', 'wb') as f:
            pickle.dump(train_data, f)

    def _create_sparse_matrix(self, rating_data: RatingData) -> TrainData:
        num_users = len(set(r.user_id for r in rating_data.data))
        num_movies = len(set(r.movie_id for r in rating_data.data))

        user_id_to_id, id_to_user_id = self._create_id_mapping([r.user_id for r in rating_data.data])
        movie_id_to_id, id_to_movie_id = self._create_id_mapping([r.movie_id for r in rating_data.data])

        rows = [user_id_to_id[r.user_id] for r in rating_data.data]
        cols = [movie_id_to_id[r.movie_id] for r in rating_data.data]
        data = [r.rating for r in rating_data.data]

        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies))

        train_data = TrainData(
            raw_data=rating_data,
            sparse_matrix=sparse_matrix,
            user_id_to_id=user_id_to_id,
            id_to_user_id=id_to_user_id,
            movie_id_to_id=movie_id_to_id,
            id_to_movie_id=id_to_movie_id
        )

        return train_data

    def process(self, rating_data: RatingData) -> TrainData:
        train_data = self._create_sparse_matrix(rating_data=rating_data)
        self._save_data(train_data=train_data)

        return train_data
