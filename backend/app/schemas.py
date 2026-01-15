from pydantic import BaseModel, EmailStr

class Register(BaseModel):
    hospital_name: str
    email: EmailStr
    contact: str
    name: str
    address: str
    username: str
    password: str

class Login(BaseModel):
    username: str
    password: str
