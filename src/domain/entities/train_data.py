from pydantic import BaseModel
from scipy.sparse import csr_matrix

class TrainData(BaseModel):
    sparse_matrix: csr_matrix
    user_id_to_id: dict[int, int]
    id_to_user_id: dict[int, int]
    movie_id_to_id: dict[int, int]
    id_to_movie_id: dict[int, int]

    class Config:
        arbitrary_types_allowed = True
