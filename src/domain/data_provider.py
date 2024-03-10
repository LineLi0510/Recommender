from fastapi import Depends

from domain.entities.movie import Movie as DomainMovieSchema
from persistence.entities.movie import Movie as PersistenceMoviesSchema
from persistence.db_service import DbService
from persistence.queries.movie_queries import query_movie_info


class MovieDataProvider:
    def __init__(self, db_service: DbService) -> None:
        self._db_service = db_service

    def _map_movie(self, movie_data: PersistenceMoviesSchema) -> DomainMovieSchema:
        """
        Map movie data to

        :param movie_data: Movie inforamtion
        :return: movie data model
        """
        movie_data = movie_data.__dict__

        return DomainMovieSchema(**movie_data)


    def get_movie_info(self, movie_id: int) -> DomainMovieSchema:
        """Test docstring for get_movie_info"""
        movie_data = query_movie_info(movie_id=movie_id, db=self._db_service.db_session)
        movie_data = self._map_movie(movie_data)

        return movie_data