from typing import Tuple

import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

from domain.entities.train_data import TrainData


def create_dataset(data_set: TrainData) -> Dataset:
    data = pd.DataFrame([(rating.user_id, rating.movie_id, rating.rating) for rating in data_set.raw_data.data])
    reader = Reader(rating_scale=(0.5, 5))
    dataset = Dataset.load_from_df(df=data, reader=reader)

    return dataset


def create_train_test_split(data_set: Dataset) -> Tuple[TrainData, TrainData]:
    train_data, test_data = train_test_split(data_set, test_size=0.2, random_state=42)

    return train_data, test_data