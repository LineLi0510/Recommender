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
    """
    Pings the service

    :return: str
    ---
    get:
        description: ping
        responses:
            200:
                content:
                    text/plain:
                    schema:
                        type: string
                        example: pong
    """
    return 'pong'
