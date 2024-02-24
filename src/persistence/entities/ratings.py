from sqlalchemy import Column, Integer, Float, DateTime

from persistence.entities.base import Base


class Ratings(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
