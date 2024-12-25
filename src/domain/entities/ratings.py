from typing import List

from pydantic import BaseModel


class Ratings(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float


class RatingData(BaseModel):
    data: List[Ratings]
