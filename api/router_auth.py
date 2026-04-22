"""
Boxify Backend — Auth Router

Endpoints:
    POST /api/auth/register
    POST /api/auth/login
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.schemas import AuthRegisterRequest, AuthLoginRequest, AuthResponse
from core.database import get_db
from core.models import User
from core.security import hash_password, verify_password, create_access_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: AuthRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken.")

    user = User(username=payload.username, hashed_password=hash_password(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"user_id": user.id, "username": user.username})
    logger.info("Registered new user: %s (id=%d)", user.username, user.id)

    return AuthResponse(access_token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=AuthResponse)
def login(payload: AuthLoginRequest, db: Session = Depends(get_db)):
    """Authenticate and receive a JWT token."""
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token({"user_id": user.id, "username": user.username})
    logger.info("User logged in: %s", user.username)

    return AuthResponse(access_token=token, user_id=user.id, username=user.username)
