from persistence.entities.movie import Movie
from persistence.db_service import Session


def query_movie_info(db: Session, movie_id: int) -> Movie:
    """
    Queries the database for a movie with the given id.

    :param db: Data base session
    :param movie_id: id of movie to query

    :return: Movie information
    """
    movie_infos = db.query(Movie).filter(Movie.movie_id == movie_id).first()

    return movie_infos


def query_all_movie_info(db: Session) -> Movie:
    """
    Queries the database for a movie with the given id.

    :param db: Data base session
    :param movie_id: id of movie to query

    :return: Movie information
    """
    movie_infos = db.query(Movie).all()

    return movie_infos