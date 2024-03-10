from fastapi import Depends

from api.main import ApiFacade
from domain.data_provider import MovieDataProvider
from persistence.database.database_setup import engine
from persistence.db_service import DbService


def create_movie_data_provider_dependency() -> MovieDataProvider:
    db_service = DbService(engine)
    return MovieDataProvider(db_service)

dependencies = {
    'movie_data_provider': MovieDataProvider(DbService(engine))
}


new_api_facade = ApiFacade(dependencies=dependencies)
app = new_api_facade.app