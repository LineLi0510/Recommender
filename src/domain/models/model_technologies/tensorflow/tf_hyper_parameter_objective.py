from typing import Dict, Any

import numpy as np
from sklearn.metrics import mean_squared_error

from domain.models.model_technologies.surprise.surprise_utils import create_train_test_split
from domain.models.model_technologies.tensorflow.tf_utils import create_train_test_split
from domain.models.model_technologies.tensorflow.tf_utils import get_algo_params, create_data_set
from src.domain.entities.train_data import TrainData


class TfHyperparameterObjective:
    def __init__(self, model_setup) -> None:
        self._model_setup = model_setup

    def objective(
            self,
            hyperparameters: Dict[str, Any],
            data_set: TrainData,
    ) -> float:
        algo_params = get_algo_params(data_set=data_set, algo_params=self._model_setup.algo_params)
        tf_data_set = create_data_set(data_set=data_set)
        train_set, test_set = create_train_test_split(data_set=tf_data_set)

        algo_class = self._model_setup.algo
        algo_params = {**algo_params, **hyperparameters}
        algo = algo_class(algo_params=algo_params)

        algo.compile(
            optimizer=algo_params["optimizer"],
            loss=algo_params["loss"],
            metrics=algo_params["metrics"]
        )

        algo.fit(
            x=train_set,
            validation_data=test_set,
            epochs=algo_params['epochs']
        )

        predictions = algo.predict([tf_data_set.user_tensor, tf_data_set.movie_tensor])
        labels_np = tf_data_set.rating_tensor.numpy()

        rmse = np.sqrt(mean_squared_error(y_true=labels_np, y_pred=np.squeeze(predictions)))

        return rmse


