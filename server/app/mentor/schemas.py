"""
Mentor System - Pydantic Schemas

Defines data models for mentor preferences, analysis requests,
insights, and recommendations.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class PersonalityStyle(str, Enum):
    """Mentor communication personality styles."""

    ENCOURAGING = "encouraging"
    PROFESSIONAL = "professional"
    CONCISE = "concise"
    EDUCATIONAL = "educational"


class ExpertiseLevel(str, Enum):
    """User's ML expertise level."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class VoiceMode(str, Enum):
    """Voice interaction preferences."""

    VOICE_FIRST = "voice_first"  # Auto-play voice
    TEXT_FIRST = "text_first"  # Show text, manual voice
    ASK_EACH_TIME = "ask_each_time"  # Prompt for each message


class SuggestionPriority(str, Enum):
    """Priority levels for mentor suggestions."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SuggestionType(str, Enum):
    """Types of mentor suggestions."""

    GREETING = "greeting"
    DATASET_ANALYSIS = "dataset_analysis"
    NEXT_STEP = "next_step"
    ERROR_EXPLANATION = "error_explanation"
    BEST_PRACTICE = "best_practice"
    LEARNING_TIP = "learning_tip"
    WARNING = "warning"


class MentorPreferences(BaseModel):
    """User's mentor configuration preferences."""

    enabled: bool = Field(True, description="Mentor feature enabled")
    avatar: str = Field("scientist", description="Selected character avatar")
    personality: PersonalityStyle = Field(
        PersonalityStyle.ENCOURAGING, description="Communication style"
    )
    voice_mode: VoiceMode = Field(VoiceMode.TEXT_FIRST, description="Voice interaction mode")
    expertise_level: ExpertiseLevel = Field(
        ExpertiseLevel.BEGINNER, description="User's ML expertise"
    )
    show_tips: bool = Field(True, description="Show learning tips")
    auto_analyze: bool = Field(True, description="Auto-analyze on dataset upload")


class DatasetInsight(BaseModel):
    """Dataset analysis insights."""

    summary: str = Field(..., description="High-level dataset summary")
    n_rows: int = Field(..., description="Number of rows")
    n_columns: int = Field(..., description="Number of columns")
    numeric_columns: List[str] = Field(default_factory=list, description="Numeric column names")
    categorical_columns: List[str] = Field(
        default_factory=list, description="Categorical column names"
    )
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Missing value counts")
    high_cardinality_columns: List[Dict[str, Any]] = Field(
        default_factory=list, description="High cardinality columns with details"
    )
    scaling_needed: List[str] = Field(default_factory=list, description="Columns needing scaling")
    recommended_target: Optional[str] = Field(None, description="Suggested target column")
    warnings: List[str] = Field(default_factory=list, description="Data quality warnings")
    recommendations: List[str] = Field(
        default_factory=list, description="Preprocessing recommendations"
    )


class MentorAction(BaseModel):
    """Actionable suggestion that mentor provides."""

    label: str = Field(..., description="Action button label")
    type: Literal["add_node", "fix_issue", "learn_more", "execute", "show_guide"] = Field(
        ..., description="Action type"
    )
    payload: Optional[Dict[str, Any]] = Field(None, description="Action metadata")


class MentorSuggestion(BaseModel):
    """Individual mentor suggestion."""

    id: str = Field(..., description="Unique suggestion ID")
    type: SuggestionType = Field(..., description="Suggestion category")
    priority: SuggestionPriority = Field(SuggestionPriority.INFO, description="Priority level")
    title: str = Field(..., description="Suggestion title")
    message: str = Field(..., description="Detailed message (supports markdown)")
    voice_text: Optional[str] = Field(None, description="Alternative text for TTS (if different)")
    actions: List[MentorAction] = Field(default_factory=list, description="Actionable buttons")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    dismissible: bool = Field(True, description="Can be dismissed by user")


class MentorAnalysisRequest(BaseModel):
    """Request for mentor to analyze current playground state."""

    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Current pipeline nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Current pipeline edges")
    dataset_metadata: Optional[Dict[str, Any]] = Field(None, description="Current dataset info")
    current_context: Optional[str] = Field(None, description="Context hint (e.g., 'error', 'idle')")
    user_message: Optional[str] = Field(None, description="Direct question from user")


class MentorGreetingRequest(BaseModel):
    """Request for personalized greeting."""

    user_name: str = Field(..., description="User's full name")
    time_of_day: Optional[str] = Field(None, description="morning/afternoon/evening")


class MentorResponse(BaseModel):
    """Complete mentor response."""

    success: bool = Field(True, description="Request successful")
    greeting: Optional[str] = Field(None, description="Personalized greeting")
    suggestions: List[MentorSuggestion] = Field(
        default_factory=list, description="List of suggestions"
    )
    dataset_insights: Optional[DatasetInsight] = Field(None, description="Dataset analysis")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next actions")
    context_summary: Optional[str] = Field(None, description="Summary of current state")


class TTSRequest(BaseModel):
    """Text-to-speech generation request."""

    text: str = Field(..., description="Text to convert to speech")
    personality: PersonalityStyle = Field(
        PersonalityStyle.ENCOURAGING, description="Voice personality"
    )
    cache_key: Optional[str] = Field(None, description="Cache key for reuse")


class TTSResponse(BaseModel):
    """Text-to-speech response."""

    success: bool = Field(..., description="Generation successful")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    duration_seconds: Optional[float] = Field(None, description="Audio duration")
    cached: bool = Field(False, description="Served from cache")
    error: Optional[str] = Field(None, description="Error message if failed")


class ErrorExplanationRequest(BaseModel):
    """Request to explain a pipeline execution error."""

    error_message: str = Field(..., description="Error message from execution")
    node_type: str = Field(..., description="Node that failed")
    node_config: Dict[str, Any] = Field(..., description="Node configuration")
    pipeline_state: Optional[Dict[str, Any]] = Field(None, description="Pipeline state")


class PipelineStepGuide(BaseModel):
    """Step-by-step guide for specific model type."""

    model_type: Literal[
        "linear_regression", "logistic_regression", "decision_tree", "random_forest"
    ] = Field(..., description="Target model type")
    steps: List[Dict[str, Any]] = Field(..., description="Ordered steps with node types")
    explanation: str = Field(..., description="Overall workflow explanation")
    estimated_time: Optional[str] = Field(None, description="Estimated completion time")
