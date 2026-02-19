"""
Gamification Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class BadgeResponse(BaseModel):
    """Single badge info."""

    badge_id: str
    name: str
    description: str
    icon: str
    color: str
    awarded_at: Optional[datetime] = None
    is_earned: bool = False

    class Config:
        from_attributes = True


class GamificationProfile(BaseModel):
    """Full gamification profile for a student."""

    xp: int = 0
    level: int = 1
    xp_to_next_level: int = 100
    progress_percent: float = 0.0
    total_badges_earned: int = 0
    badges: List[BadgeResponse] = []


class AwardXPRequest(BaseModel):
    """Request to award XP for an action."""

    action: str = Field(..., description="Action that triggered XP award")
    context: Optional[dict] = Field(None, description="Additional context for badge checking")


class AwardXPResponse(BaseModel):
    """Response after awarding XP."""

    xp_gained: int
    total_xp: int
    level: int
    leveled_up: bool
    new_level: Optional[int] = None
    new_badges: List[BadgeResponse] = []
