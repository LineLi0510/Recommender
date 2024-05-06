from typing import Union

from pydantic import BaseModel
from surprise import KNNBasic
from surprise import NormalPredictor
from surprise import SVD

AlgoT = Union[NormalPredictor, SVD, KNNBasic]


svd = {
    'model_name': 'SVD',
    'algo': SVD(),
    'model_saving_path': '../data/models/surprise/svd.pkl'
}

knn_base = {
    'model_name': 'knn_base',
    'algo': KNNBasic(sim_options={
        'name': 'cosine',
        'user_based': False
    }),
    'model_saving_path': '../data/models/surprise/knn_base.pkl'
}

knn_als = {
    'model_name': 'knn_als',
    'algo': KNNBasic(
        bsl_options={
            'method': 'als',
            'n_epochs': 20,
        },
        sim_options={
            'name': 'pearson_baseline'
        }),
    'model_saving_path': '../data/models/surprise/knn_als.pkl'
}

normal_predictor = {
    'model_name': 'normal_predictor',
    'algo': NormalPredictor(),
    'model_saving_path': '../data/models/surprise/normal_predictor.pkl'
}

models_to_train = [svd, knn_base, knn_als, normal_predictor]
