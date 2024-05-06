import pandas as pd
from domain.model_training.data_loader.raw_data_loader import RawDataLoader
from domain.model_training.data_processing.train_data_creator import TrainDataCreator
from domain.model_training.model.surprise.Model import SurpriseModel, SurpriseModelCV
from domain.model_training.model.surprise.model_training_pipeline import models_to_train
from domain.model_training.train_config import TrainConfig
from persistence.database.database_setup import engine
from persistence.db_service import DbService
from persistence.entities.base import Base

Base.metadata.create_all(engine)
chunksize = 10000


def run_training():
    db_service = DbService(engine)
    raw_data = RawDataLoader(train_config=TrainConfig, db_service=db_service).process()
    # raw_data = pd.read_csv('../data/ratings.csv')
    # train_data = TrainDataCreator(train_config=TrainConfig).process(raw_data)
    SurpriseModelCV().process(ratings=raw_data, model_configs=models_to_train)
    SurpriseModel().process(ratings=raw_data, model_configs=models_to_train)


if __name__ == '__main__':
    run_training()
