from sqlalchemy.orm import Session

from persistence.entities.ratings import Ratings


def query_average_user_ratings(db: Session, user_id: str):
    movie_infos = db.query(Ratings).filter(Ratings.user_id == user_id).first()

    return movie_infos


def query_user_ratings(db: Session, user_id: int):
    movie_infos = db.query(Ratings).filter(Ratings.user_id == user_id).all()

    return movie_infos


def query_movie_ratings(db: Session, movie_id: int):
    movie_infos = db.query(Ratings).filter(Ratings.movie_id == movie_id).all()

    return movie_infos


def query_all_ratings(db: Session):
    movie_infos = db.query(Ratings).all()

    return movie_infos