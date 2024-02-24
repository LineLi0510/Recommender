from persistence.entities.movies import Movies


def query_movie_info(db, movie_id: int):
    movie_infos = db.query(Movies).filter(Movies.movie_id == movie_id).first()

    return movie_infos