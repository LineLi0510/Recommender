from typing import Dict, Any

from tensorflow.keras import Model, layers, regularizers

from domain.models.model_technologies.tensorflow.tf_algos.abstract_algo import AbstractTfAlgo


class TfL2RegAlgo(AbstractTfAlgo):
    def __init__(self, algo_params: Dict[str, Any]) -> None:
        super().__init__(algo_params=algo_params)

    def _init_model(self) -> Model:
        user_input = layers.Input(shape=(1,), name="user")
        item_input = layers.Input(shape=(1,), name="item")

        user_embedding = layers.Embedding(
            input_dim=self.algo_params['num_users'],
            output_dim=self.algo_params['embedding_dim'],
            embeddings_regularizer=regularizers.l2(self.algo_params["l2_regularization"])
        )(user_input)

        item_embedding = layers.Embedding(
            input_dim=self.algo_params['num_items'],
            output_dim=self.algo_params['embedding_dim'],
            embeddings_regularizer=regularizers.l2(self.algo_params["l2_regularization"])
        )(item_input)

        user_embedding = layers.Dropout(self.algo_params["dropout"])(user_embedding)
        item_embedding = layers.Dropout(self.algo_params["dropout"])(item_embedding)

        dot_product = layers.Dot(axes=-1)([user_embedding, item_embedding])
        output = layers.Dense(1, activation=self.algo_params["activation"])(dot_product)

        model = Model(inputs=[user_input, item_input], outputs=output)

        return model
