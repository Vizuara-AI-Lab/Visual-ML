"""
SQLAlchemy database configuration and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Create engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Dependency to get database session.

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    from app.db.base import Base

    Base.metadata.create_all(bind=engine)
