import pandas as pd
import numpy as np

from persistence.database.database_setup import Session


def import_dataframe_to_database(orm_class, data: pd.DataFrame) -> None:
    session = Session()
    for _, row in data.iterrows():
        cleaned_row = {key: float(value) if isinstance(value, np.float64) else value for key, value in row.items()}
        obj = orm_class(**cleaned_row)
        session.add(obj)

    session.commit()


def delete_table(engine, table) -> None:
    table.__table__.drop(engine)


def query_data_as_dataframe(query, engine) -> pd.DataFrame:
    df = pd.read_sql_query(sql=query, con=engine)

    return df


def load_data_as_dataframe(table_name: str, engine) -> pd.DataFrame:
    df = pd.read_sql_table(table_name, con=engine)

    return df
