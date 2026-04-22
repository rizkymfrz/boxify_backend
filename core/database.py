"""
Boxify Backend — Database Configuration

Sets up the SQLAlchemy engine and declarative base.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Expects MYSQL_URL environment variable, falling back to a local test db if not present
# Example: mysql+pymysql://user:password@localhost:3306/boxify_db
DATABASE_URL = os.getenv("MYSQL_URL", "mysql+pymysql://samtek_user:samtek123@localhost:3306/boxify")

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
