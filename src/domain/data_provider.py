from domain.entities.Movie import Movie
from persistence.db_service import DbService
from persistence.queries.movie_queries import query_movie_info


class MovieDataProvider:
    def __init__(self, db_service: DbService) -> None:
        self._db_service = db_service

    def get_movie_info(self, movie_id: int) -> Movie:
        movie_data = query_movie_info(db=self._db_service.db, movie_id=movie_id)
        movie_response = Movie(
            movie_id=movie_data.movie_id,
            title=movie_data.title,
            genres=movie_data.genres
        )

        return movie_response