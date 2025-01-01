from typing import TypeVar

from surprise import KNNBasic
from surprise import SVD

from domain.models.hyperparameter_tuning_technologies.optuna_hyperparameter_tuner import OptunaHyperparameterTuner
from domain.models.model_technologies.surprise.surprise_cross_validator import SurpriseCrossValidator
from domain.models.model_technologies.surprise.surprise_model import SurpriseRecommenderModel
from domain.models.model_technologies.tensorflow.tf_algos.base_algo import TfBaseAlgo
from domain.models.model_technologies.tensorflow.tf_algos.dropout_algo import TfL2RegAlgo
from domain.models.model_technologies.tensorflow.tf_model import TensorFlowRecommenderModel

mlf_tracking_uri: str = 'http://mlflow:5001'

ModelType = TypeVar("ModelType", SurpriseRecommenderModel, TensorFlowRecommenderModel)
AlgoType = TypeVar("AlgoType", KNNBasic, SVD, TfBaseAlgo, TfL2RegAlgo)
ValidatorType = TypeVar("ValidatorType", bound=SurpriseCrossValidator)