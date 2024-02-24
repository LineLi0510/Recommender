from fastapi import Depends

from api.main import ApiFacade
from domain.data_provider import MovieDataProvider
from persistence.database.database_setup import engine
from persistence.db_service import DbService

db_service = DbService(engine)
movie_data_provider = MovieDataProvider(db_service=db_service)

def create_movie_data_provider_dependency() -> MovieDataProvider:
    return MovieDataProvider(db_service=db_service)


new_api_facade = ApiFacade(dependencies=[Depends(create_movie_data_provider_dependency)])
app = new_api_facade.app