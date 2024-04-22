from domain.model_training.data_loader.raw_data_loader import RawDataLoader
from domain.model_training.data_processing.train_data_creator import TrainDataCreator
from persistence.database.database_setup import engine
from persistence.db_service import DbService
from persistence.entities.base import Base

Base.metadata.create_all(engine)
chunksize = 10000

def run_training():
    db_service = DbService(engine)
    raw_data = RawDataLoader(db_service).process()
    train_data = TrainDataCreator().process(raw_data)



if __name__ == '__main__':
    run_training()
