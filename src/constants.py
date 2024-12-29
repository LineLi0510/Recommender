from typing import TypeVar

from surprise import KNNBasic
from surprise import SVD

from domain.models.hyperparameter_tuning_technologies.optuna_hyperparameter_tuner import OptunaHyperparameterTuner
from domain.models.model_technologies.surprise.surprise_cross_validator import SurpriseCrossValidator
from domain.models.model_technologies.surprise.surprise_model import SurpriseRecommenderModel


mlf_tracking_uri: str = 'http://mlflow:5001'

ModelType = TypeVar("ModelType", bound=SurpriseRecommenderModel)
AlgoType = TypeVar("AlgoType", KNNBasic, SVD)
TunerType = TypeVar("TunerType", bound=OptunaHyperparameterTuner)
ValidatorType = TypeVar("ValidatorType", bound=SurpriseCrossValidator)