from datetime import datetime

from pydantic import BaseModel


class Ratings(BaseModel):
    id: int
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime
