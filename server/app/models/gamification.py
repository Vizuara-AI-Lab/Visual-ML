"""
Gamification models: StudentBadge for tracking earned badges.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.session import Base


class StudentBadge(Base):
    """Tracks badges earned by students."""

    __tablename__ = "student_badges"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False, index=True)
    badge_id = Column(String(100), nullable=False)  # matches key in BADGES dict
    awarded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    student = relationship("Student", back_populates="badges")

    def __repr__(self):
        return f"<StudentBadge(student={self.student_id}, badge={self.badge_id})>"
