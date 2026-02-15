"""
AI Mentor API Router

Provides endpoints for:
- Personalized greetings
- Dataset intelligence analysis
- Pipeline state analysis
- Contextual suggestions
- Error explanations
- Text-to-speech generation
- User preference management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import json

from app.mentor.schemas import (
    MentorGreetingRequest,
    MentorAnalysisRequest,
    MentorResponse,
    MentorPreferences,
    TTSRequest,
    TTSResponse,
    ErrorExplanationRequest,
    PipelineStepGuide,
    PersonalityStyle,
    ExpertiseLevel,
)
from app.mentor.services.dataset_analyzer import dataset_analyzer
from app.mentor.services.pipeline_analyzer import pipeline_analyzer
from app.mentor.services.recommendation_engine import recommendation_engine
from app.mentor.services.tts_service import tts_service
from app.core.security import get_current_student
from app.db.session import get_db
from app.models.user import Student
from app.core.logging import logger

router = APIRouter(prefix="/mentor", tags=["AI Mentor"])


@router.post("/greet", response_model=MentorResponse)
async def greet_user(
    request: MentorGreetingRequest,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Generate personalized greeting for user.

    Returns welcome message with quick action suggestions.
    """
    try:
        # Get user preferences
        preferences = _get_user_preferences(student, db)

        # Generate greeting
        greeting_suggestion = recommendation_engine.generate_greeting(
            user_name=request.user_name,
            personality=preferences.personality,
            time_of_day=request.time_of_day,
        )

        return MentorResponse(
            success=True, greeting=greeting_suggestion.message, suggestions=[greeting_suggestion]
        )

    except Exception as e:
        logger.error(f"Error generating greeting: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate greeting: {str(e)}",
        )


@router.post("/analyze-dataset", response_model=MentorResponse)
async def analyze_dataset(
    request: MentorAnalysisRequest,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Analyze dataset and provide intelligent insights.

    Returns data quality assessment, preprocessing recommendations,
    and warnings about potential issues.
    """
    try:
        if not request.dataset_metadata:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset metadata required for analysis",
            )

        # Get user preferences
        preferences = _get_user_preferences(student, db)

        # Generate dataset suggestions
        suggestions = recommendation_engine.generate_dataset_suggestions(
            dataset_metadata=request.dataset_metadata,
            personality=preferences.personality,
            expertise_level=preferences.expertise_level,
        )

        # Extract dataset insights from suggestions
        dataset_insights = None
        for suggestion in suggestions:
            if suggestion.actions and len(suggestion.actions) > 0:
                for action in suggestion.actions:
                    if action.type == "learn_more" and action.payload:
                        insights_data = action.payload.get("insights")
                        if insights_data:
                            dataset_insights = insights_data
                            break

        return MentorResponse(
            success=True,
            suggestions=suggestions,
            dataset_insights=dataset_insights,
            context_summary=f"Analyzed dataset with {request.dataset_metadata.get('n_rows', 0)} rows "
            f"and {request.dataset_metadata.get('n_columns', 0)} columns",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze dataset: {str(e)}",
        )


@router.post("/analyze-pipeline", response_model=MentorResponse)
async def analyze_pipeline(
    request: MentorAnalysisRequest,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Analyze current pipeline state and suggest improvements.

    Detects missing steps, validates connections, and recommends next actions.
    """
    try:
        # Get user preferences
        preferences = _get_user_preferences(student, db)

        # Generate contextual suggestions
        suggestions = recommendation_engine.generate_contextual_suggestions(
            nodes=request.nodes,
            edges=request.edges,
            dataset_metadata=request.dataset_metadata,
            personality=preferences.personality,
            expertise_level=preferences.expertise_level,
        )

        # Analyze pipeline structure
        pipeline_analysis = pipeline_analyzer.analyze_pipeline(
            request.nodes, request.edges, request.dataset_metadata
        )

        return MentorResponse(
            success=True,
            suggestions=suggestions,
            next_steps=pipeline_analysis.get("next_steps", []),
            context_summary=pipeline_analysis.get("message", "Pipeline analyzed"),
        )

    except Exception as e:
        logger.error(f"Error analyzing pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze pipeline: {str(e)}",
        )


@router.post("/explain-error", response_model=MentorResponse)
async def explain_error(
    request: ErrorExplanationRequest,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Explain execution error in friendly terms with fix suggestions.
    """
    try:
        # Get user preferences
        preferences = _get_user_preferences(student, db)

        # Generate error explanation
        error_suggestion = recommendation_engine.explain_error(
            error_message=request.error_message,
            node_type=request.node_type,
            node_config=request.node_config,
            personality=preferences.personality,
        )

        return MentorResponse(
            success=True,
            suggestions=[error_suggestion],
            context_summary=f"Error in {request.node_type}: {request.error_message[:100]}",
        )

    except Exception as e:
        logger.error(f"Error explaining error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain error: {str(e)}",
        )


@router.post("/generate-speech", response_model=TTSResponse)
async def generate_speech(request: TTSRequest, student: Student = Depends(get_current_student)):
    """
    Generate speech audio from text using Inworld TTS.
    """
    try:
        logger.info(f"📢 TTS Request from {student.emailId}")
        logger.info(f"   Text: {request.text[:100]}...")
        logger.info(f"   Personality: {request.personality}")

        response = await tts_service.generate_speech(
            text=request.text,
            personality=request.personality,
            cache_key=request.cache_key,
            return_base64=True,
        )

        logger.info(
            f"📢 TTS Response: success={response.success}, has_audio={bool(response.audio_base64)}, cached={response.cached}"
        )
        if not response.success:
            logger.error(f"📢 TTS Error: {response.error}")
        elif response.audio_base64:
            logger.info(f"📢 TTS Audio size: {len(response.audio_base64)} chars (base64)")

        return response

    except Exception as e:
        logger.error(f"❌ TTS endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate speech: {str(e)}",
        )


@router.get("/test-tts")
async def test_tts():
    """Test TTS without authentication for debugging."""
    try:
        response = await tts_service.generate_speech(
            text="Hello, this is a test of the text to speech system.",
            personality=PersonalityStyle.ENCOURAGING,
            return_base64=True,
        )
        return {
            "success": response.success,
            "has_audio": bool(response.audio_base64),
            "audio_length": len(response.audio_base64) if response.audio_base64 else 0,
            "cached": response.cached,
            "error": response.error,
        }
    except Exception as e:
        logger.error(f"Test TTS failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@router.get("/preferences", response_model=MentorPreferences)
async def get_preferences(
    student: Student = Depends(get_current_student), db: Session = Depends(get_db)
):
    """Get user's mentor preferences."""
    try:
        return _get_user_preferences(student, db)
    except Exception as e:
        logger.error(f"Error getting preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get preferences: {str(e)}",
        )


@router.put("/preferences", response_model=MentorPreferences)
async def update_preferences(
    preferences: MentorPreferences,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """Update user's mentor preferences."""
    try:
        # Get the actual database student object
        db_student = db.query(Student).filter(Student.id == student.id).first()

        if not db_student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found",
            )

        # Store preferences in user model (recentProject field as JSON)
        # This is a temporary solution - ideally add a dedicated mentor_preferences column
        db_student.recentProject = json.dumps(preferences.model_dump())
        db.commit()
        db.refresh(db_student)

        logger.info(f"Updated mentor preferences for user {student.id}")
        return preferences

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating preferences: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preferences: {str(e)}",
        )


@router.post("/get-guide/{model_type}", response_model=PipelineStepGuide)
async def get_model_guide(model_type: str, student: Student = Depends(get_current_student)):
    """
    Get step-by-step guide for specific model type.
    """
    try:
        guides = {
            "linear_regression": PipelineStepGuide(
                model_type="linear_regression",
                steps=[
                    {
                        "step": 1,
                        "node_type": "upload_file",
                        "description": "Upload your dataset with numeric features",
                    },
                    {
                        "step": 2,
                        "node_type": "missing_value_handler",
                        "description": "Handle any missing values",
                    },
                    {
                        "step": 3,
                        "node_type": "encoding",
                        "description": "Encode categorical columns (if any)",
                    },
                    {
                        "step": 4,
                        "node_type": "scaling",
                        "description": "Scale features for better performance (optional)",
                    },
                    {
                        "step": 5,
                        "node_type": "split",
                        "description": "Split into training and test sets",
                    },
                    {
                        "step": 6,
                        "node_type": "linear_regression",
                        "description": "Train Linear Regression model",
                    },
                    {
                        "step": 7,
                        "node_type": "metrics",
                        "description": "Evaluate with R², RMSE, MAE",
                    },
                ],
                explanation="Linear Regression predicts continuous numeric values. It works best with "
                "linear relationships between features and target.",
                estimated_time="5-10 minutes",
            ),
            "logistic_regression": PipelineStepGuide(
                model_type="logistic_regression",
                steps=[
                    {
                        "step": 1,
                        "node_type": "upload_file",
                        "description": "Upload dataset with binary/categorical target",
                    },
                    {
                        "step": 2,
                        "node_type": "missing_value_handler",
                        "description": "Clean missing values",
                    },
                    {
                        "step": 3,
                        "node_type": "encoding",
                        "description": "Encode categorical features",
                    },
                    {"step": 4, "node_type": "split", "description": "Split data (80/20 or 70/30)"},
                    {
                        "step": 5,
                        "node_type": "logistic_regression",
                        "description": "Train classification model",
                    },
                    {
                        "step": 6,
                        "node_type": "confusion_matrix",
                        "description": "Visualize prediction results",
                    },
                ],
                explanation="Logistic Regression classifies data into categories. Best for binary classification tasks.",
                estimated_time="5-10 minutes",
            ),
            "decision_tree": PipelineStepGuide(
                model_type="decision_tree",
                steps=[
                    {"step": 1, "node_type": "upload_file", "description": "Upload your dataset"},
                    {
                        "step": 2,
                        "node_type": "missing_value_handler",
                        "description": "Handle missing values",
                    },
                    {
                        "step": 3,
                        "node_type": "encoding",
                        "description": "Encode categorical variables",
                    },
                    {"step": 4, "node_type": "split", "description": "Create train/test split"},
                    {"step": 5, "node_type": "decision_tree", "description": "Train decision tree"},
                    {
                        "step": 6,
                        "node_type": "feature_importance",
                        "description": "View important features",
                    },
                ],
                explanation="Decision Trees create interpretable rules for predictions. Works for both "
                "classification and regression.",
                estimated_time="5-10 minutes",
            ),
        }

        if model_type not in guides:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Guide not found for model type: {model_type}",
            )

        return guides[model_type]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model guide: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model guide: {str(e)}",
        )


@router.post("/model-introduction/{model_type}", response_model=MentorResponse)
async def get_model_introduction(
    model_type: str,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get detailed introduction and explanation for a specific model type.

    This provides an interactive explanation with dataset preparation options.
    """
    try:
        preferences = _get_user_preferences(student, db)

        introduction_suggestion = recommendation_engine.generate_model_introduction(
            model_type=model_type,
            personality=preferences.personality,
            expertise_level=preferences.expertise_level,
        )

        return MentorResponse(
            success=True,
            suggestions=[introduction_suggestion],
            context_summary=f"Introduction to {model_type}",
        )

    except Exception as e:
        logger.error(f"Error generating model introduction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate introduction: {str(e)}",
        )


@router.post("/dataset-guidance", response_model=MentorResponse)
async def get_dataset_guidance(
    request: dict,
    student: Student = Depends(get_current_student),
    db: Session = Depends(get_db),
):
    """
    Get guidance for dataset preparation based on user's choice.
    """
    try:
        preferences = _get_user_preferences(student, db)

        action = request.get("action")
        model_type = request.get("model_type")
        next_message = request.get("next_message", "")

        if not action or not model_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="action and model_type are required",
            )

        guidance_suggestion = recommendation_engine.generate_dataset_guidance(
            action=action,
            model_type=model_type,
            next_message=next_message,
            personality=preferences.personality,
        )

        return MentorResponse(
            success=True,
            suggestions=[guidance_suggestion],
            context_summary=f"Dataset guidance for {model_type}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating dataset guidance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate guidance: {str(e)}",
        )


# Helper functions


def _get_user_preferences(student: Student, db: Session) -> MentorPreferences:
    """Extract mentor preferences from user model."""
    try:
        # Get the actual database student object
        db_student = db.query(Student).filter(Student.id == student.id).first()

        if not db_student:
            logger.warning(f"Student {student.id} not found in database")
            return MentorPreferences()

        if db_student.recentProject:
            # Try to parse as JSON (mentor preferences)
            try:
                prefs_data = json.loads(db_student.recentProject)
                # Check if it's mentor preferences or just regular bio text
                if isinstance(prefs_data, dict) and "personality" in prefs_data:
                    return MentorPreferences(**prefs_data)
            except json.JSONDecodeError:
                pass  # Not JSON, use defaults

        # Return default preferences
        return MentorPreferences()
    except Exception as e:
        logger.warning(f"Error parsing user preferences: {str(e)}")
        return MentorPreferences()
