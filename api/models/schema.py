from pydantic import BaseModel, EmailStr, Field

from typing import List, Optional

from fastapi import UploadFile, File
class UserCreate(BaseModel):
    email: EmailStr  
    name: str
    password: str



class UserLogin(BaseModel):
    email: EmailStr
    password: str
    

class OutlierSchema(BaseModel):
    genes: list[str]
    


class AnnotationSchema(BaseModel):
    organism_name: str
    id_type: str
