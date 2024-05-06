from pydantic import BaseModel


class TrainConfig(BaseModel):
    raw_data: str  = "../data/"
    training_data: str = '../data/train_data/ratings_train.pkl'