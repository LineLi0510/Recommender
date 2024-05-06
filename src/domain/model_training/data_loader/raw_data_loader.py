from typing import List

from domain.data_provider import RatingDataProvider
from domain.entities.ratings import Ratings
from domain.model_training.train_config import TrainConfig
from persistence.db_service import DbService


class RawDataLoader:
    def __init__(self, train_config: TrainConfig, db_service: DbService):
        self.db_service = db_service

    def process(self) -> List[Ratings]:
        rating_data_provider = RatingDataProvider(db_service=self.db_service)
        rating_data = rating_data_provider.get_all_rating_info()

        return rating_data
