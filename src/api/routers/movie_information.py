from typing import Annotated

from fastapi import APIRouter, Depends

from api.entities.movie_information import Movie
from domain.data_provider import MovieDataProvider

movie_router = APIRouter()


@movie_router.get("/movie/{movie_id}", response_model=None)
def get_movie_info(movie_id: int,  movie_data_provider: MovieDataProvider = Depends()) -> str:
    movie_data = movie_data_provider.get_movie_info(movie_id=movie_id)

    return 'hello'
