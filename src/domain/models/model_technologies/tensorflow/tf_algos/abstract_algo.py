from abc import ABC, abstractmethod
from typing import Dict, Any, Literal, List

from pydantic import BaseModel
from tensorflow.keras import Model, layers


"""
Alog-Erklärung

- Embedding-Layer:
    - Layer um kategoriale Variablen in kontinuierliche, dichte Werte zu übersetzen (jede Kategorie wird auf einen Vekor 
    im kontinuierlichen Raum abgebildet)
    - Dimension des Layers ist (dim_input, dim_output)
    - Ist ein trainierbarer Layer, der kategoriale Daten in kontinuierliche, niedrigdimensionale Vektoren übersetzt
- L2 regularisierung in den Embedding layer
    - Embeddings werden bestraft, die sehr hohe Werte annehmen
    - Hilft Overfitting zu vermeiden
- Dropout nach den Embedding-Schichten
    - Neuronen werden zufällig während des Trainings deativiert
    - Hilft so ebenfalls Overfitting zu vermeiden
- Dot-Schicht:
    - Ermittelt Skalarprodukt zwischen den Embeddings für User und Movie
    - Modeliert damit die Ähnlichkeit  / Interaktion  zwischen User und Item
    - Output ist Skalar, der Fit ausdrückt
- Dense-Schicht
    - Transformation des Rohwertes aus der Dot-Schicht auf eine Vorhersage wie Wahrscheinlichkeit oder Rating (Abhängig 
    von Aktivierungsfunktion)
"""


class TfAlgoParams(BaseModel):
    num_users: int
    num_items: int
    embedding_dim: int
    optimizer: Literal["adam"]
    loss: Literal["mse"]
    metrics: List[str]


class AbstractTfAlgo(ABC, Model):
    def __init__(self, algo_params: Dict[str, Any]) -> None:
        super().__init__()
        self.algo_params = algo_params
        self.algo = self._init_model()

    @abstractmethod
    def _init_model(self) -> Model:
        pass

    def call(self, inputs):
        return self.algo(inputs)
