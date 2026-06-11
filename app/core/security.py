from datetime import datetime,timedelta,timezone
from authlib.jose import jwt,JWTError
from app.core.config import Settings
from fastapi import HTTPException


def create_token(data:dict, expire_minutes=30):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc)+ timedelta(minutes=expire_minutes)
    to_encode.update({'exp':expire})
    jwt.encode(
        to_encode,
        Settings.JWT_SECRET_KEY,
        algorithm = Settings.JWT_ALGORITHM
    )

def verify_token(token:str):
    try:
        payload = jwt.decode(
            token,
            Settings.JWT_SECRET_KEY,
            algorithm=[Settings.JWT_ALGORITHM])
        return payload
       
    
    except JWTError:
        return None



