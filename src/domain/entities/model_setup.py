from typing import Type, Optional, Any, Dict, Generic

from pydantic import BaseModel

from constants import ModelType, AlgoType, ValidatorType
from domain.models.hyperparameter_tuning_technologies.optuna_hyperparameter_tuner import OptunaHyperparameterTuner


class ModelSetup(BaseModel, Generic[ModelType, AlgoType, ValidatorType]):
    class Config:
        arbitrary_types_allowed = True

    model_name: str
    model_class: Type[ModelType]
    model_path: str
    algo: Type[AlgoType] | None = None
    algo_params: Dict[str, Any]
    hyperparameter_tuner: Optional[OptunaHyperparameterTuner] = None
    hyperparameter_objective: Optional[type] = None
    hyperparameter_tuning_config: Optional[dict] = None
    cross_validator: Type[ValidatorType] | None = None
    cross_validation_params: Optional[dict] = None
