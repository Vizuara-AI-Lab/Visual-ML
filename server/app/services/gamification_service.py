"""
Gamification service — handles XP awards, level calculation, and badge checking.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models.user import Student
from app.models.gamification import StudentBadge
from app.services.badge_definitions import BADGES, LEVEL_THRESHOLDS, XP_AWARDS
from app.schemas.gamification import BadgeResponse, GamificationProfile, AwardXPResponse
from app.core.logging import logger


def calculate_level(xp: int) -> int:
    """Calculate level from total XP using thresholds."""
    level = 1
    for i, threshold in enumerate(LEVEL_THRESHOLDS):
        if xp >= threshold:
            level = i + 1
        else:
            break
    return level


def xp_for_next_level(current_xp: int) -> int:
    """Calculate XP needed to reach the next level."""
    current_level = calculate_level(current_xp)
    if current_level >= len(LEVEL_THRESHOLDS):
        return 0  # Max level reached
    return LEVEL_THRESHOLDS[current_level] - current_xp


def level_progress_percent(current_xp: int) -> float:
    """Calculate progress percentage toward next level."""
    current_level = calculate_level(current_xp)
    if current_level >= len(LEVEL_THRESHOLDS):
        return 100.0

    current_threshold = LEVEL_THRESHOLDS[current_level - 1]
    next_threshold = LEVEL_THRESHOLDS[current_level]
    range_xp = next_threshold - current_threshold

    if range_xp == 0:
        return 100.0

    progress = current_xp - current_threshold
    return round((progress / range_xp) * 100, 1)


def get_earned_badge_ids(db: Session, student_id: int) -> set:
    """Get set of badge_ids already earned by this student."""
    rows = db.query(StudentBadge.badge_id).filter(
        StudentBadge.student_id == student_id
    ).all()
    return {row[0] for row in rows}


def award_badge(db: Session, student_id: int, badge_id: str) -> Optional[BadgeResponse]:
    """Award a badge to a student. Returns BadgeResponse if newly awarded, None if already had it."""
    badge_def = BADGES.get(badge_id)
    if not badge_def:
        return None

    # Check if already earned
    existing = db.query(StudentBadge).filter(
        StudentBadge.student_id == student_id,
        StudentBadge.badge_id == badge_id,
    ).first()

    if existing:
        return None

    new_badge = StudentBadge(student_id=student_id, badge_id=badge_id)
    db.add(new_badge)
    db.flush()

    logger.info(f"Badge awarded: {badge_id} to student {student_id}")

    return BadgeResponse(
        badge_id=badge_id,
        name=badge_def["name"],
        description=badge_def["description"],
        icon=badge_def["icon"],
        color=badge_def["color"],
        awarded_at=new_badge.awarded_at,
        is_earned=True,
    )


def check_and_award_badges(
    db: Session,
    student_id: int,
    action: str,
    context: Optional[Dict[str, Any]] = None,
) -> List[BadgeResponse]:
    """Check badge conditions and award any newly earned badges."""
    new_badges: List[BadgeResponse] = []
    earned = get_earned_badge_ids(db, student_id)

    # first_pipeline: on first pipeline execution
    if action == "pipeline_execution" and "first_pipeline" not in earned:
        badge = award_badge(db, student_id, "first_pipeline")
        if badge:
            new_badges.append(badge)

    # welcome: on first login
    if action == "first_login" and "welcome" not in earned:
        badge = award_badge(db, student_id, "welcome")
        if badge:
            new_badges.append(badge)

    # app_creator: on first app published
    if action == "app_published" and "app_creator" not in earned:
        badge = award_badge(db, student_id, "app_creator")
        if badge:
            new_badges.append(badge)

    # story_complete: on first story completed
    if action == "story_completed" and "story_complete" not in earned:
        badge = award_badge(db, student_id, "story_complete")
        if badge:
            new_badges.append(badge)

    # perfect_score: when context includes r2 > 0.95
    if context and context.get("r2_score", 0) > 0.95 and "perfect_score" not in earned:
        badge = award_badge(db, student_id, "perfect_score")
        if badge:
            new_badges.append(badge)

    return new_badges


def award_xp(
    db: Session,
    student_id: int,
    action: str,
    context: Optional[Dict[str, Any]] = None,
) -> AwardXPResponse:
    """Award XP to a student and check for level ups and badges."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return AwardXPResponse(
            xp_gained=0, total_xp=0, level=1, leveled_up=False
        )

    xp_amount = XP_AWARDS.get(action, 10)  # Default 10 XP for unknown actions
    old_level = student.level

    student.xp = (student.xp or 0) + xp_amount
    student.level = calculate_level(student.xp)

    leveled_up = student.level > old_level

    # Check for new badges
    new_badges = check_and_award_badges(db, student_id, action, context)

    db.commit()

    if leveled_up:
        logger.info(f"Student {student_id} leveled up: {old_level} -> {student.level}")

    return AwardXPResponse(
        xp_gained=xp_amount,
        total_xp=student.xp,
        level=student.level,
        leveled_up=leveled_up,
        new_level=student.level if leveled_up else None,
        new_badges=new_badges,
    )


def get_gamification_profile(db: Session, student_id: int) -> GamificationProfile:
    """Get full gamification profile for a student."""
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        return GamificationProfile()

    current_xp = student.xp or 0
    earned_ids = get_earned_badge_ids(db, student_id)

    # Build badge list with earned status
    all_badges = []
    for badge_id, badge_def in BADGES.items():
        is_earned = badge_id in earned_ids
        # Get awarded_at if earned
        awarded_at = None
        if is_earned:
            sb = db.query(StudentBadge).filter(
                StudentBadge.student_id == student_id,
                StudentBadge.badge_id == badge_id,
            ).first()
            if sb:
                awarded_at = sb.awarded_at

        all_badges.append(BadgeResponse(
            badge_id=badge_id,
            name=badge_def["name"],
            description=badge_def["description"],
            icon=badge_def["icon"],
            color=badge_def["color"],
            awarded_at=awarded_at,
            is_earned=is_earned,
        ))

    return GamificationProfile(
        xp=current_xp,
        level=student.level or 1,
        xp_to_next_level=xp_for_next_level(current_xp),
        progress_percent=level_progress_percent(current_xp),
        total_badges_earned=len(earned_ids),
        badges=all_badges,
    )
