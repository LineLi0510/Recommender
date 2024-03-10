from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker

db_url: str = "mysql+mysqlconnector://user:user_password@mysql/my_database"
engine: Engine = create_engine(db_url, echo=True)
Session = sessionmaker(bind=engine)
