from sqlalchemy import Column, Integer, String
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    hospital_name = Column(String)
    email = Column(String, unique=True)
    contact = Column(String)
    name = Column(String)
    address = Column(String)
    username = Column(String, unique=True)
    password = Column(String)
