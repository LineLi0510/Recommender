from typing import Dict, Any, Tuple

import tensorflow as tf
from pydantic import BaseModel

from domain.entities.train_data import TrainData


class TfDataSet(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    data_set: tf.data.Dataset
    user_tensor: tf.Tensor
    movie_tensor: tf.Tensor
    rating_tensor: tf.Tensor


def get_algo_params(algo_params: Dict[str, Any], data_set: TrainData):
    algo_params['num_users'] = len(set(data_set.id_to_user_id.keys()))
    algo_params['num_items'] = len(set(data_set.id_to_movie_id.keys()))

    return algo_params

def create_data_set(data_set: TrainData) -> TfDataSet:
    user_tensor, movie_tensor, rating_tensor = _preprocess_train_data(data_set)

    full_dataset = tf.data.Dataset.from_tensor_slices(
        ({
             "user": user_tensor,
             "item": movie_tensor
         }, rating_tensor)
    )

    return TfDataSet(
        user_tensor=user_tensor,
        movie_tensor=movie_tensor,
        rating_tensor=rating_tensor,
        data_set=full_dataset
    )

def create_train_test_split(data_set: TfDataSet) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_size = int(0.8 * len(data_set.rating_tensor))

    train_data = data_set.data_set.take(train_size).batch(32)
    test_data = data_set.data_set.skip(train_size).batch(32)

    return train_data, test_data


def _preprocess_train_data(data: TrainData) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Wandelt die TrainData in TensorFlow-kompatible Arrays um.

    :param data: TrainData-Instanz mit den Trainingsdaten
    :return: Benutzer-IDs, Item-IDs, Bewertungen (alle als Tensoren)
    """
    raw_data = data.raw_data.data

    user_tensor = tf.constant([data.user_id_to_id[entry.user_id] for entry in raw_data], dtype=tf.int32)
    movie_tensor = tf.constant([data.movie_id_to_id[entry.movie_id] for entry in raw_data], dtype=tf.int32)
    rating_tensor = tf.constant([entry.rating for entry in raw_data], dtype=tf.float32)

    return user_tensor, movie_tensor, rating_tensor
