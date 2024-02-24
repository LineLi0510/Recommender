from pydantic import BaseModel


class RequestTitleRecommendation(BaseModel):
    name: str
    age: int
    gender: str
