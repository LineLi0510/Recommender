from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_url = "mysql+mysqlconnector://user:user_password@mysql/my_database"
engine = create_engine(db_url, echo=True)
Session = sessionmaker(bind=engine)
