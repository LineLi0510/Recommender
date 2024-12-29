from typing import Type, Optional, Any, Dict, Generic

from pydantic import BaseModel

from constants import ModelType, AlgoType, TunerType, ValidatorType


class ModelSetup(BaseModel, Generic[ModelType, AlgoType, TunerType, ValidatorType]):
    class Config:
        arbitrary_types_allowed = True

    model_name: str
    model_class: Type[ModelType]
    model_path: str
    algo: Type[AlgoType]
    algo_params: Dict[str, Any]
    hyperparameter_tuner: Optional[Type[TunerType]] = None
    hyperparameter_objective: Optional[type] = None
    hyperparameter_tuning_config: Optional[dict] = None
    cross_validator: Type[ValidatorType]
    cross_validation_params: Optional[dict] = None
