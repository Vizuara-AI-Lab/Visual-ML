"""
Gamification API endpoints — XP, levels, and badges.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.security import get_current_student
from app.models.user import Student
from app.schemas.gamification import GamificationProfile, AwardXPRequest, AwardXPResponse
from app.services import gamification_service

router = APIRouter(prefix="/gamification", tags=["Gamification"])


@router.get("/profile", response_model=GamificationProfile)
async def get_profile(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Get the current student's gamification profile (XP, level, badges)."""
    return gamification_service.get_gamification_profile(db, student.id)


@router.post("/award-xp", response_model=AwardXPResponse)
async def award_xp(
    request: AwardXPRequest,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Award XP to the current student for completing an action."""
    return gamification_service.award_xp(
        db, student.id, request.action, request.context
    )


@router.get("/badges")
async def get_all_badges(
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Get all badges with the current student's earned status."""
    profile = gamification_service.get_gamification_profile(db, student.id)
    return {"badges": profile.badges}
