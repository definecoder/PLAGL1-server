from fastapi import FastAPI
# from sqlalchemy.orm import Session
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from jose import JWTError, jwt
# from datetime import datetime, timedelta, timezone
# from passlib.context import CryptContext
# from api.models.models import User
# from database import SessionLocal, engine
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# from database import get_db
# from core.security import hash_password, verify_password

from routers.auth_router import router as auth_router
from routers.operation_router import router as operation_router

app = FastAPI()

origins = [
    "http://localhost:3000",  
    "https://yourfrontenddomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins from the list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(auth_router)
app.include_router(operation_router)

# Authenticate the user


# Create access token








