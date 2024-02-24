from pydantic import BaseModel


class RequestUserRecommendation(BaseModel):
    user: str
    age: int
    gender: str
