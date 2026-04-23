"""
Boxify Backend — Database Configuration

Sets up the SQLAlchemy engine and declarative base.
All connection settings are loaded from core.config (sourced from .env).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependency to yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
