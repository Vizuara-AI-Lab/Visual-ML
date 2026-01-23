"""
SQLAlchemy database configuration and session management.
Production-grade with connection pooling for scalability.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Create engine with production-grade connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    
    # Connection Pool Configuration
    pool_size=settings.DB_POOL_SIZE,          # Max persistent connections (20)
    max_overflow=settings.DB_MAX_OVERFLOW,    # Allow extra connections (10)
    pool_timeout=settings.DB_POOL_TIMEOUT,    # Wait time for connection (30s)
    pool_pre_ping=True,                        # Test connection before use
    pool_recycle=settings.DB_POOL_RECYCLE,    # Recycle after 1 hour
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
