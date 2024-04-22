from typing import List

from domain.entities.movie import Movie as DomainMovieSchema
from domain.entities.ratings import Ratings as DomainRatingsSchema
from persistence.db_service import DbService
from persistence.entities.movie import Movie as PersistenceMoviesSchema
from persistence.entities.ratings import Ratings as PerictenceRatingsSchema
from persistence.queries.movie_queries import query_movie_info, query_all_movie_info
from persistence.queries.ratings_queries import query_all_ratings, query_user_ratings, query_movie_ratings


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

    def get_all_movie_info(self) -> DomainMovieSchema:
        """Test docstring for get_movie_info"""
        movie_data = query_all_movie_info(db=self._db_service.db_session)
        movie_data = [self._map_movie(movie) for movie in movie_data]

        return movie_data


class RatingDataProvider:
    def __init__(self, db_service: DbService) -> None:
        self._db_service = db_service

    def _map_ratings(self, rating_data: PerictenceRatingsSchema) -> DomainRatingsSchema:
        """
        Map movie data to

        :param rating_data: Rating inforamtion
        :return: rating data model
        """
        rating_data = rating_data.__dict__

        return DomainRatingsSchema(**rating_data)


    def get_rating_info_for_movie(self, movie_id: int) -> List[DomainRatingsSchema]:
        """Test docstring for get_movie_info"""
        rating_data = query_movie_ratings(movie_id=movie_id, db=self._db_service.db_session)
        rating_data = [self._map_ratings(rating) for rating in rating_data]

        return rating_data


    def get_rating_info_for_user(self, user_id: int) -> List[DomainRatingsSchema]:
        """Test docstring for get_movie_info"""
        rating_data = query_user_ratings(user_id=user_id, db=self._db_service.db_session)
        rating_data = [self._map_ratings(rating) for rating in rating_data]

        return rating_data

    def get_all_rating_info(self) -> List[DomainRatingsSchema]:
        """Test docstring for get_movie_info"""
        rating_data = query_all_ratings(db=self._db_service.db_session)
        rating_data = [self._map_ratings(rating) for rating in rating_data]

        return rating_data