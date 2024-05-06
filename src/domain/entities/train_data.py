from pydantic import BaseModel
from scipy.sparse import csr_matrix
from typing import Dict

class TrainData(BaseModel):
    sparse_matrix: csr_matrix
    user_id_to_id: Dict[int, int]
    id_to_user_id: Dict[int, int]
    movie_id_to_id: Dict[int, int]
    id_to_movie_id: Dict[int, int]

    class Config:
        arbitrary_types_allowed = True
