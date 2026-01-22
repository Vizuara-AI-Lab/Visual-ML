"""
Email tasks for background processing using Celery.
"""

from app.core.celery_app import celery_app
from app.services.email_service import email_service
from app.core.logging import logger


@celery_app.task(name="send_welcome_email")
def send_welcome_email_task(email: str, name: str):
    """
    Send welcome email immediately after verification.
    
    Args:
        email: User email
        name: User name
    """
    try:
        success = email_service.send_welcome_email(email, name)
        if success:
            logger.info(f"Welcome email sent to {email}")
        else:
            logger.error(f"Failed to send welcome email to {email}")
        return success
    except Exception as e:
        logger.error(f"Error sending welcome email to {email}: {str(e)}")
        raise


@celery_app.task(name="send_experience_survey")
def send_experience_survey_task(email: str, name: str):
    """
    Send experience survey 10 days after signup.
    
    Args:
        email: User email
        name: User name
    """
    try:
        success = email_service.send_experience_survey(email, name)
        if success:
            logger.info(f"Experience survey sent to {email}")
        else:
            logger.error(f"Failed to send experience survey to {email}")
        return success
    except Exception as e:
        logger.error(f"Error sending experience survey to {email}: {str(e)}")
        raise


@celery_app.task(name="send_marketing_email")
def send_marketing_email_task(email: str, name: str, subject: str, content: str):
    """
    Send marketing/promotional email.
    
    Args:
        email: User email
        name: User name
        subject: Email subject
        content: HTML email content
    """
    try:
        success = email_service.send_marketing_email(email, name, subject, content)
        if success:
            logger.info(f"Marketing email sent to {email}")
        else:
            logger.error(f"Failed to send marketing email to {email}")
        return success
    except Exception as e:
        logger.error(f"Error sending marketing email to {email}: {str(e)}")
        raise
