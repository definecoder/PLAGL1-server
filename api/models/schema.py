from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr  
    name: str
    password: str



class UserLogin(BaseModel):
    email: EmailStr
    password: str
    

class OutlierSchema(BaseModel):
    genes: list[str]
    