from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker

db_url: str = "mysql+mysqlconnector://user:user_password@localhost:3306/my_database"
# db_url: str = "mysql+mysqlconnector://user:user_password@mysql/my_database"
# db_url: str = "mysql://root:user_password@localhost:3306/rec_test"
engine: Engine = create_engine(db_url, echo=True)
Session = sessionmaker(bind=engine)
