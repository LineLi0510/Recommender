from fastapi import FastAPI

from api.routers import movie_information
from api.routers import ping

class ApiFacade:
    def __init__(self, dependencies):
        self._app = FastAPI()
        self._dependencies = dependencies
        self._app.include_router(movie_information.movie_router, dependencies=self._dependencies)
        self._app.include_router(ping.ping_router)

    @property
    def app(self):
        return self._app
