from sqlalchemy import Column, Integer, String

from persistence.entities.base import Base


class Movies(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True, autoincrement=True)
    movie_id = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    genres = Column(String(255), nullable=False)
