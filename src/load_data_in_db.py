import logging
from typing import List

import pandas as pd
from tqdm import tqdm

from persistence.database.database_setup import engine, Session
from persistence.database.database_utils import delete_table
from persistence.database.database_utils import import_dataframe_to_database
from persistence.entities.base import Base
from persistence.entities.movie import Movie
from persistence.entities.ratings import Ratings

logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)


Base.metadata.create_all(engine)
chunksize = 10000


def import_csv_to_mysql(path: str, orm_class: Base, columns_date_format: List[str] = []):
    session = Session()
    try:
        data_chunks = pd.read_csv(path, chunksize=chunksize)
        for chunk in tqdm(data_chunks, desc='importing chunks', unit='chunk'):
            chunk.rename(columns={'movieId': 'movie_id', 'userId': 'user_id'}, inplace=True)
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = chunk[col].apply(lambda x: float(x))
            for col in columns_date_format:
                chunk = chunk.drop(columns=[col], axis=1)
            import_dataframe_to_database(orm_class=orm_class, data=chunk, session=session)

        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def delete_given_table(table) -> None:
    delete_table(engine=engine, table=table)


if __name__ == '__main__':
    import_csv_to_mysql(path='data/movies.csv', orm_class=Movie)
    import_csv_to_mysql(path='data/ratings.csv', orm_class=Ratings, columns_date_format=['timestamp'])
    # delete_given_table(table=Ratings)

    # db_service = DbService(engine)
    # movie_data_provider = MovieDataProvider(db_service=db_service)
    # movie_data = movie_data_provider.get_movie_info(movie_id=5)
    # print(movie_data)
