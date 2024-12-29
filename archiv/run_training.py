from datetime import datetime

from domain.model_training.data_loader.raw_data_loader import RawDataLoader
from domain.model_training.data_processing.train_data_creator import TrainDataCreator
from domain.model_training.train_config import TrainConfig
from archiv.model_training_pipeline import models_to_train
from persistence.database.database_setup import engine
from persistence.db_service import DbService
from persistence.entities.base import Base
from src.domain.utils.ml_flow_tracker import MlFlowTracker


Base.metadata.create_all(engine)
chunksize = 100000

train_models = False
run_cross_validation = False
run_hyperparametertuning = True


def run_training():
    db_service = DbService(engine)
    raw_data = RawDataLoader(db_service=db_service).process()
    data_set = TrainDataCreator(train_config=TrainConfig()).process(raw_data)

    if train_models:
        for model_config in models_to_train:
            tracker = MlFlowTracker(experiment_name=f'training/{model_config.model_name}')
            tracker.start_run(run_name=f'{model_config.model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}')
            model = model_config.model_class.create_model_from_dataset(
                data_set=data_set,
                algo=model_config.algo,
                algo_params=model_config.algo_params,
                model_path=model_config.model_path)
            model_kpis = model.calculate_model_metrics()
            model.dump_model()
            tracker.log_params(model_config.algo_params)
            tracker.log_metrics(model_kpis)
            tracker.end_run()

    if run_cross_validation:
        for model_config in models_to_train:
            tracker = MlFlowTracker(experiment_name=f'cross_validation/{model_config.model_name}')
            tracker.start_run(run_name=f'{model_config.model_name}_{datetime.now().strftime("%Y%m%d-%H%M")}')
            cross_validator = model_config.cross_validator(
                algo=model_config.algo,
                algo_params=model_config.algo_params,
                cross_validation_setup=model_config.cross_validation_params
            )
            model_kpis = cross_validator.cross_validate_model(
                data_set=data_set,
            )
            tracker.log_cv_metrics(model_kpis)
            tracker.end_run()

    if run_hyperparametertuning:
        for model_config in models_to_train:
            if model_config.hyperparameter_tuner:
                if not model_config.hyperparameter_objective:
                    raise ValueError('No objective function defined for hyperparameter tuning')
                model_hyperparameter_tuner = model_config.hyperparameter_tuner()
                model_hyperparameter_tuner.run_hyperparameter_tuning(
                    model_setup=model_config,
                    data_set=data_set,
                    objective_method=model_config.hyperparameter_objective(model_setup=model_config).objective
                )


if __name__ == '__main__':
    run_training()
