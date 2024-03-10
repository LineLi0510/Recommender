from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from domain.entities.movie import Movie
from domain.data_provider import MovieDataProvider

movie_router = APIRouter()


@movie_router.get("/movie/{movie_id}", response_model=None)
def get_movie_info(movie_id: str,  movie_data_provider: MovieDataProvider = Depends(MovieDataProvider)) -> Movie:
    """
    Retrieve information

    :param movie_id: Movie id to query information
    :param movie_data_provider: Movie data
    :return: Str with movie information
    ---
    get:
        description: Retrieve movie information
        parameters:
        - in: path
            name: movie_id
            schema:
                type: int
                required: true
        response:
            200:
                description: Movie data to given movie id
                schema:
                    type: object
    """

    movie_data = movie_data_provider.get_movie_info(movie_id=int(movie_id))

    return movie_data
