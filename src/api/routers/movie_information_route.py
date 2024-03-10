from fastapi import APIRouter

from domain.entities.movie import Movie

movie_router = APIRouter()

class MovieInformationRoute:
    def __init__(self, movie_data_provider):
        self.movie_data_provider = movie_data_provider
        self.router = APIRouter()
        self.router.add_api_route(
            "/movie/{movie_id}",
            self.get_movie_information,
            methods=["GET"],
            response_model=Movie,
            summary="Get movie",
            description="Retrieve information about a movie id",
            tags=["movies"]
        )

    def get_movie_information(self, movie_id: int) -> Movie:
        movie_data = self.movie_data_provider.get_movie_info(movie_id=int(movie_id))

        return movie_data
