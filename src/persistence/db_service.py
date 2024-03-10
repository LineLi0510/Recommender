from typing import Optional

from sqlalchemy.orm import Session


class DbService:
    def __init__(self, engine) -> None:
        self._db: Optional[Session] = None
        self._engine = engine

    @property
    def db_session(self) -> Session:
        """Returns the database session"""
        if not self._db:
            self._db = Session(self._engine)

        return self._db