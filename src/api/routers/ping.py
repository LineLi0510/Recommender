from fastapi import APIRouter

ping_router = APIRouter()

db_config = {
    "host": "mysql",
    "user": "user",
    "password": "user_password",
    "database": "my_database"
}

@ping_router.get('/ping', response_model=None)
def root() -> str:
    return 'ping'
