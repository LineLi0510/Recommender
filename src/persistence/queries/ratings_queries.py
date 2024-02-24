from persistence.entities.ratings import Ratings


def query_average_user_ratings(db, user_id: str):
    movie_infos = db.query(Ratings).filter(Ratings.user_id == user_id).first()

    return movie_infos


def query_user_ratings(db, user_id: str):
    movie_infos = db.query(Ratings).filter(Ratings.user_id == user_id).s()

    return movie_infos