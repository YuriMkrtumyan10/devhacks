from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import jwt
import bcrypt
from pydantic import BaseModel, EmailStr

from models import User, Company
from database import get_db
import os

router = APIRouter(tags=["authentication"])

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    company_name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def hash_password(password: str) -> str:
    """Hash a password"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

@router.post("/login", response_model=TokenResponse)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""

    # Find user by email
    user = db.query(User).filter(User.email == login_data.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Verify password
    if not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    # Get user's company info
    company = db.query(Company).filter(Company.id == user.company_id).first()

    # Prepare user data
    user_data = {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "companyId": user.company_id,
        "company": {
            "id": company.id,
            "name": company.name,
            "industry": company.industry,
            "country": company.country
        } if company else None
    }

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_data
    }

@router.post("/register", response_model=TokenResponse)
def register(register_data: RegisterRequest, db: Session = Depends(get_db)):
    """Register new user and company"""

    # Check if user already exists
    existing_user = db.query(User).filter(User.email == register_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    try:
        # Create company first
        company = Company(
            name=register_data.company_name,
            contact_email=register_data.email
        )
        db.add(company)
        db.flush()  # Flush to get the company ID

        # Hash password
        hashed_password = hash_password(register_data.password)

        # Create user
        user = User(
            email=register_data.email,
            password_hash=hashed_password,
            role="user",  # First user of company is owner
            company_id=company.id
        )
        db.add(user)
        db.commit()

        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )

        # Prepare user data
        user_data = {
            "id": user.id,
            "email": user.email,
            "role": user.role,
            "companyId": user.company_id,
            "company": {
                "id": company.id,
                "name": company.name,
                "industry": company.industry,
                "country": company.country
            }
        }

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_data
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    """Logout user (client should remove token)"""
    return {"message": "Successfully logged out"}

@router.get("/verify")
def verify_token(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Verify if token is valid and return user info"""

    # Get user's company info
    company = db.query(Company).filter(Company.id == current_user.company_id).first()

    user_data = {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "companyId": current_user.company_id,
        "company": {
            "id": company.id,
            "name": company.name,
            "industry": company.industry,
            "country": company.country
        } if company else None
    }

    return {"user": user_data}
