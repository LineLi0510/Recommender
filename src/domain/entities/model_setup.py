from typing import Union, Type, Optional, Any, Dict

from pydantic import BaseModel
from surprise import KNNBasic
from surprise import SVD

from domain.models.model_technologies.surprise.surprise_cross_validator import SurpriseCrossValidator
from domain.models.model_technologies.surprise.surprise_hyper_parameter_objective import SurpriseHyperparameterObjective
from domain.models.model_technologies.surprise.surprise_model import SurpriseRecommenderModel
from domain.models.model_technologies.tf.tf_model import TensorFlowRecommenderModel
from domain.models.hyperparameter_tuning_technologies.optuna_hyperparameter_tuner import OptunaHyperparameterTuner


class AbstractModelSetup(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

    model_name: str
    model_class: Union[Type[SurpriseRecommenderModel], Type[TensorFlowRecommenderModel]]
    model_path: str
    algo: Union[Type[KNNBasic], Type[SVD]]
    algo_params: Dict[str, Any]
    cross_validator: Union[Type[SurpriseCrossValidator]]
    cross_validation_params: Optional[dict] = None
    hyperparameter_tuner: Optional[type[OptunaHyperparameterTuner]] = None
    hyperparameter_objective: Optional[type[SurpriseHyperparameterObjective]] = None
    hyperparameter_tuning_config: Optional[dict] = None


class SurpriseModelSetup(AbstractModelSetup):
    model_class: Type[SurpriseRecommenderModel]
    algo: Union[Type[KNNBasic], Type[SVD]]
    cross_validator: Union[Type[SurpriseCrossValidator]]
    hyperparameter_objective: Optional[type[SurpriseHyperparameterObjective]] = None
