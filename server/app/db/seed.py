"""
Database seeding script.
Creates default admin user for initial setup.
"""

from sqlalchemy.orm import Session
from app.db.session import engine, Base, SessionLocal
from app.models.user import Admin, UserRole
from app.core.security import hash_password
from app.core.logging import logger
import os


def seed_database():
    """Seed database with initial data."""

    # Create all tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    try:
        # Check if admin exists
        existing_admin = db.query(Admin).filter(Admin.email == "admin@visualml.com").first()

        if not existing_admin:
            # Create default admin
            admin = Admin(
                email="admin@visualml.com",
                password=hash_password("Admin123"),  # Change this in production!
                role=UserRole.ADMIN,
                name="System Administrator",
                isActive=True,
            )
            db.add(admin)
            db.commit()
            logger.info("✅ Default admin created:")
            logger.info("   Email: admin@visualml.com")
            logger.info("   Password: Admin123")
            logger.info("   ⚠️  IMPORTANT: Change this password in production!")
        else:
            logger.info("Admin user already exists, skipping...")

        logger.info("✅ Database seeding completed!")

    except Exception as e:
        logger.error(f"Error seeding database: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
