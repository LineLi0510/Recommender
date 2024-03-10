from typing import Dict, Any

from fastapi import FastAPI

from api.routers import ping
from api.routers.movie_information_route import MovieInformationRoute


class ApiFacade:
    def __init__(self, dependencies: Dict[str, Any]):
        self._app = FastAPI()

        movie_information_route = MovieInformationRoute(movie_data_provider=dependencies['movie_data_provider'])
        self._app.include_router(movie_information_route.router)

        self._app.include_router(ping.ping_router)

    @property
    def app(self):
        return self._app
