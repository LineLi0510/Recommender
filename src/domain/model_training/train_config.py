from pydantic import BaseModel


class DataProcessingConfig(BaseModel):
    raw_data: str  = "../data/"
    training_data: str = '../data/train_data/ratings_train.pkl'