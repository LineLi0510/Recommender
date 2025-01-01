from surprise import SVD, KNNBasic

from domain.entities.model_setup import ModelSetup
from domain.models.hyperparameter_tuning_technologies.optuna_hyperparameter_tuner import OptunaHyperparameterTuner
from domain.models.model_technologies.surprise.surprise_cross_validator import SurpriseCrossValidator
from domain.models.model_technologies.surprise.surprise_hyper_parameter_objective import SurpriseHyperparameterObjective
from domain.models.model_technologies.surprise.surprise_model import SurpriseRecommenderModel
from domain.models.model_technologies.tensorflow.tf_algos.base_algo import TfBaseAlgo
from domain.models.model_technologies.tensorflow.tf_algos.dropout_algo import TfL2RegAlgo
from domain.models.model_technologies.tensorflow.tf_hyper_parameter_objective import TfHyperparameterObjective
from domain.models.model_technologies.tensorflow.tf_model import TensorFlowRecommenderModel

surprise_svd = ModelSetup(
    model_name = 'SVD',
    model_class = SurpriseRecommenderModel,
    model_path='/data/models/surprise/svd.pkl',
    algo=SVD,
    algo_params={},
    cross_validator=SurpriseCrossValidator,
    cross_validation_params={
        'measures': ['RMSE', 'MAE'],
        'cv': 5},
    hyperparameter_tuner=OptunaHyperparameterTuner(trials=20),
    hyperparameter_objective=SurpriseHyperparameterObjective,
    hyperparameter_tuning_config={
        'n_factors': {'type': 'int', 'low': 20, 'high': 500},
        'n_epochs': {'type': 'int', 'low':5, 'high': 100},
        'lr_all': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-1},
        'reg_all': {'type': 'loguniform', 'low': 1e-4, 'high': 1e-1}
    }
)


surprise_knn = ModelSetup(
    model_name="KNN",
    model_class = SurpriseRecommenderModel,
    model_path='/data/models/surprise/knn_base.pkl',
    algo=KNNBasic,
    algo_params={
        'sim_options': {
            'name': 'cosine',
            'user_based': False
        }
    },
    cross_validator=SurpriseCrossValidator,
    cross_validation_params={
        'measures': ['RMSE', 'MAE'],
        'cv': 5},
    hyperparameter_tuner=OptunaHyperparameterTuner(trials=20),
    hyperparameter_objective=SurpriseHyperparameterObjective,
    hyperparameter_tuning_config={
        'k': {'type': 'int', 'low': 10, 'high': 100},  # Anzahl der Nachbarn
        'sim_options.name': {'type': 'categorical', 'values': ['cosine', 'msd', 'pearson']},  # Ähnlichkeitsmetrik
        'sim_options.user_based': {'type': 'categorical', 'values': [True, False]},  # User-based oder Item-based
        'min_k': {'type': 'int', 'low': 1, 'high': 10}  # Minimale Anzahl von Nachbarn
    }
)


surprise_knn_als = ModelSetup(
    model_name="KNN_ALS",
    model_class = SurpriseRecommenderModel,
    model_path='/data/models/surprise/knn_als.pkl',
    algo=KNNBasic,
    algo_params={
            'bsl_options':{
                'method': 'als',
                'n_epochs': 20,
            },
            'sim_options': {
                'name': 'pearson_baseline'
            }
        },
    cross_validator=SurpriseCrossValidator,
    cross_validation_params={
        'measures': ['RMSE', 'MAE'],
        'cv': 5},
    hyperparameter_tuning_config=None
    )

tf_base = ModelSetup(
    model_name="tf_base",
    model_class = TensorFlowRecommenderModel,
    model_path='/data/models/surprise/tf.pkl',
    algo=TfBaseAlgo,
    algo_params={
        'embedding_dim': 32,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae'],
        'activation': "relu",
        'epochs': 15

    },
    cross_validator=None,
    cross_validation_params={},
    hyperparameter_tuner=OptunaHyperparameterTuner(trials=20),
    hyperparameter_objective=TfHyperparameterObjective,
    hyperparameter_tuning_config= {
        'embedding_dim': {'type': 'int', 'low': 16, 'high': 128},
        'optimizer': {'type': 'categorical', 'values': ["adam", "sgd", "rmsprop"]},
        'activation': {'type': 'categorical', 'values': ["relu", "tanh", "sigmoid"]},
        'epochs': {'type': 'int', 'low': 5, 'high': 15}
    }
    )

tf_regularized = ModelSetup(
    model_name="tf_regularized",
    model_class = TensorFlowRecommenderModel,
    model_path='/data/models/surprise/tf_regularized.pkl',
    algo=TfL2RegAlgo,
    algo_params={
        'embedding_dim': 32,
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['mae'],
        'activation': "relu",
        'epochs': 15,
        'dropout': 0.2,
        'l2_regularization': 1e-6

    },
    cross_validator=None,
    cross_validation_params={},
    hyperparameter_tuner=OptunaHyperparameterTuner(trials=20),
    hyperparameter_objective=TfHyperparameterObjective,
    hyperparameter_tuning_config= {
        'embedding_dim': {'type': 'int', 'low': 16, 'high': 128},
        'optimizer': {'type': 'categorical', 'values': ["adam", "sgd", "rmsprop"]},
        'activation': {'type': 'categorical', 'values': ["relu", "tanh", "sigmoid"]},
        'l2_regularization': {'type': 'float', 'low': 1e-6, 'high': 1e-3},
        'epochs': {'type': 'int', 'low': 5, 'high': 15}
    }
    )



models_to_train = [tf_base, tf_regularized]
