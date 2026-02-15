"""
Recommendation Engine

Main intelligence hub that generates contextual mentor suggestions
by combining dataset analysis, pipeline analysis, and user context.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from app.mentor.schemas import (
    MentorSuggestion,
    SuggestionType,
    SuggestionPriority,
    PersonalityStyle,
    MentorAction,
    ExpertiseLevel,
)
from app.mentor.services.dataset_analyzer import dataset_analyzer
from app.mentor.services.pipeline_analyzer import pipeline_analyzer
from app.mentor.guides import LINEAR_REGRESSION_GUIDE
from app.core.logging import logger


class RecommendationEngine:
    """Generate intelligent mentor recommendations based on context."""

    def generate_greeting(
        self,
        user_name: str,
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
        time_of_day: Optional[str] = None,
    ) -> MentorSuggestion:
        """Generate personalized greeting."""

        # Time-based greeting
        if time_of_day == "morning":
            time_greeting = "Good morning"
        elif time_of_day == "afternoon":
            time_greeting = "Good afternoon"
        elif time_of_day == "evening":
            time_greeting = "Good evening"
        else:
            time_greeting = "Hello"

        # Personality-based messages
        greetings = {
            PersonalityStyle.ENCOURAGING: (
                f"{time_greeting}, {user_name}! 🎉 I'm your AI mentor, here to guide you through building amazing "
                "machine learning models. What would you like to create today?"
            ),
            PersonalityStyle.PROFESSIONAL: (
                f"{time_greeting}, {user_name}. I'm ready to assist you in developing your ML pipeline. "
                "Please specify your project objective."
            ),
            PersonalityStyle.CONCISE: (
                f"{time_greeting}, {user_name}. Ready to build ML models. What's your goal?"
            ),
            PersonalityStyle.EDUCATIONAL: (
                f"{time_greeting}, {user_name}! I'm here to teach you machine learning while we build together. "
                "Let's start by choosing what type of problem you want to solve."
            ),
        }

        message = greetings.get(personality, greetings[PersonalityStyle.ENCOURAGING])

        return MentorSuggestion(
            id=str(uuid.uuid4()),
            type=SuggestionType.GREETING,
            priority=SuggestionPriority.INFO,
            title="Welcome to ML Playground",
            message=message,
            actions=[
                MentorAction(
                    label="Linear Regression",
                    type="show_guide",
                    payload={"model_type": "linear_regression"},
                ),
                MentorAction(
                    label="Logistic Regression",
                    type="show_guide",
                    payload={"model_type": "logistic_regression"},
                ),
                MentorAction(
                    label="Decision Tree",
                    type="show_guide",
                    payload={"model_type": "decision_tree"},
                ),
                MentorAction(
                    label="Upload Dataset", type="add_node", payload={"node_type": "upload_file"}
                ),
            ],
            timestamp=datetime.utcnow().isoformat(),
            dismissible=True,
        )

    def generate_contextual_suggestions(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        dataset_metadata: Optional[Dict[str, Any]],
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
        expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER,
    ) -> List[MentorSuggestion]:
        """Generate suggestions based on current playground state."""

        suggestions = []

        # Analyze pipeline
        pipeline_analysis = pipeline_analyzer.analyze_pipeline(nodes, edges, dataset_metadata)

        # Generate gap suggestions
        if pipeline_analysis.get("gaps"):
            for gap in pipeline_analysis["gaps"]:
                priority = (
                    SuggestionPriority.WARNING
                    if "requires" in gap.lower() or "no" in gap.lower()
                    else SuggestionPriority.INFO
                )

                suggestions.append(
                    MentorSuggestion(
                        id=str(uuid.uuid4()),
                        type=(
                            SuggestionType.WARNING
                            if priority == SuggestionPriority.WARNING
                            else SuggestionType.NEXT_STEP
                        ),
                        priority=priority,
                        title="Pipeline Gap Detected",
                        message=gap,
                        actions=self._generate_gap_actions(gap),
                        timestamp=datetime.utcnow().isoformat(),
                    )
                )

        # Generate next step suggestions
        if pipeline_analysis.get("next_steps"):
            message = self._personalize_message(
                "Here's what I recommend next:\n\n"
                + "\n".join(f"• {step}" for step in pipeline_analysis["next_steps"][:3]),
                personality,
                expertise_level,
            )

            suggestions.append(
                MentorSuggestion(
                    id=str(uuid.uuid4()),
                    type=SuggestionType.NEXT_STEP,
                    priority=SuggestionPriority.INFO,
                    title="Recommended Next Steps",
                    message=message,
                    actions=self._generate_next_step_actions(pipeline_analysis["next_steps"]),
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        # Add warnings from pipeline analysis
        if pipeline_analysis.get("warnings"):
            for warning in pipeline_analysis["warnings"]:
                suggestions.append(
                    MentorSuggestion(
                        id=str(uuid.uuid4()),
                        type=SuggestionType.WARNING,
                        priority=SuggestionPriority.WARNING,
                        title="Pipeline Configuration Issue",
                        message=warning,
                        timestamp=datetime.utcnow().isoformat(),
                    )
                )

        return suggestions

    def generate_dataset_suggestions(
        self,
        dataset_metadata: Dict[str, Any],
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
        expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER,
    ) -> List[MentorSuggestion]:
        """Generate suggestions based on dataset analysis."""

        suggestions = []

        # Analyze dataset
        insights = dataset_analyzer.analyze_dataset(
            dataset_id=dataset_metadata.get("dataset_id", "unknown"),
            columns=dataset_metadata.get("columns", []),
            dtypes=dataset_metadata.get("dtypes", {}),
            n_rows=dataset_metadata.get("n_rows", 0),
            n_columns=dataset_metadata.get("n_columns", 0),
            missing_values=dataset_metadata.get("missing_values", {}),
            statistics=dataset_metadata.get("statistics"),
            preview_data=dataset_metadata.get("preview_data"),
        )

        # Dataset summary suggestion
        summary_message = self._personalize_message(
            f"**Dataset Analysis Complete!**\n\n{insights.summary}\n\n"
            f"**Key Findings:**\n"
            f"• Numeric features: {len(insights.numeric_columns)}\n"
            f"• Categorical features: {len(insights.categorical_columns)}\n"
            f"• Recommended target: {insights.recommended_target or 'Please specify'}",
            personality,
            expertise_level,
        )

        suggestions.append(
            MentorSuggestion(
                id=str(uuid.uuid4()),
                type=SuggestionType.DATASET_ANALYSIS,
                priority=SuggestionPriority.INFO,
                title="Dataset Analysis Complete",
                message=summary_message,
                actions=[
                    MentorAction(
                        label="View Details",
                        type="learn_more",
                        payload={"insights": insights.model_dump()},
                    )
                ],
                timestamp=datetime.utcnow().isoformat(),
            )
        )

        # High cardinality warnings
        if insights.high_cardinality_columns:
            for col_info in insights.high_cardinality_columns[:2]:  # Limit to 2
                suggestions.append(
                    MentorSuggestion(
                        id=str(uuid.uuid4()),
                        type=SuggestionType.WARNING,
                        priority=SuggestionPriority.WARNING,
                        title=f"High Cardinality: {col_info['column']}",
                        message=col_info["warning"],
                        actions=[
                            MentorAction(
                                label="Use Label Encoding",
                                type="add_node",
                                payload={"node_type": "encoding", "method": "label"},
                            )
                        ],
                        timestamp=datetime.utcnow().isoformat(),
                    )
                )

        # Preprocessing recommendations
        if insights.recommendations:
            rec_message = "**Recommended Preprocessing Steps:**\n\n" + "\n".join(
                f"• {rec}" for rec in insights.recommendations[:4]
            )

            suggestions.append(
                MentorSuggestion(
                    id=str(uuid.uuid4()),
                    type=SuggestionType.BEST_PRACTICE,
                    priority=SuggestionPriority.INFO,
                    title="Preprocessing Recommendations",
                    message=rec_message,
                    actions=[
                        MentorAction(
                            label="Add Missing Value Handler",
                            type="add_node",
                            payload={"node_type": "missing_value_handler"},
                        ),
                        MentorAction(
                            label="Add Encoding", type="add_node", payload={"node_type": "encoding"}
                        ),
                    ],
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

        return suggestions

    def explain_error(
        self,
        error_message: str,
        node_type: str,
        node_config: Dict[str, Any],
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
    ) -> MentorSuggestion:
        """Generate friendly error explanation with fix suggestions."""

        # Parse common error patterns
        explanation, actions = self._parse_error(error_message, node_type, node_config)

        # Personalize explanation
        if personality == PersonalityStyle.ENCOURAGING:
            prefix = "Don't worry, this is a common issue! "
        elif personality == PersonalityStyle.PROFESSIONAL:
            prefix = "Error identified. "
        elif personality == PersonalityStyle.CONCISE:
            prefix = ""
        else:
            prefix = "Let's learn from this error. "

        message = prefix + explanation

        return MentorSuggestion(
            id=str(uuid.uuid4()),
            type=SuggestionType.ERROR_EXPLANATION,
            priority=SuggestionPriority.CRITICAL,
            title=f"Error in {node_type.replace('_', ' ').title()}",
            message=message,
            actions=actions,
            timestamp=datetime.utcnow().isoformat(),
        )

    def _parse_error(
        self, error_message: str, node_type: str, node_config: Dict[str, Any]
    ) -> tuple[str, List[MentorAction]]:
        """Parse error and generate explanation with actions."""

        error_lower = error_message.lower()

        # Target column errors
        if "target column" in error_lower or "target_column" in error_lower:
            return (
                "The model needs to know which column to predict (the target). "
                "Make sure you've added a Split node before this model and specified the target column.",
                [
                    MentorAction(
                        label="Add Split Node", type="add_node", payload={"node_type": "split"}
                    ),
                    MentorAction(
                        label="Learn About Targets",
                        type="learn_more",
                        payload={"topic": "target_column"},
                    ),
                ],
            )

        # Dataset not found errors
        if "dataset" in error_lower and ("not found" in error_lower or "empty" in error_lower):
            return (
                "The node couldn't find the dataset. This usually means the previous node didn't execute successfully. "
                "Make sure all nodes before this one have executed without errors.",
                [
                    MentorAction(
                        label="Check Pipeline",
                        type="fix_issue",
                        payload={"action": "validate_pipeline"},
                    )
                ],
            )

        # Categorical encoding errors
        if "categorical" in error_lower or "non-numeric" in error_lower or "encode" in error_lower:
            return (
                "Machine learning models need numeric inputs, but your data contains text/categorical columns. "
                "Add an Encoding node before training to convert these to numbers.",
                [
                    MentorAction(
                        label="Add Encoding Node",
                        type="add_node",
                        payload={"node_type": "encoding"},
                    )
                ],
            )

        # Missing values errors
        if "missing" in error_lower or "nan" in error_lower or "null" in error_lower:
            return (
                "Your dataset has missing values (empty cells). "
                "Add a Missing Value Handler node to clean your data before training.",
                [
                    MentorAction(
                        label="Add Missing Value Handler",
                        type="add_node",
                        payload={"node_type": "missing_value_handler"},
                    )
                ],
            )

        # Default fallback
        return (
            f"An error occurred: {error_message}\n\n"
            "Check your node configuration and ensure all previous nodes executed successfully.",
            [
                MentorAction(
                    label="View Documentation", type="learn_more", payload={"node_type": node_type}
                )
            ],
        )

    def _generate_gap_actions(self, gap: str) -> List[MentorAction]:
        """Generate actions to fill pipeline gaps."""
        actions = []

        gap_lower = gap.lower()

        if "upload" in gap_lower or "data source" in gap_lower:
            actions.append(
                MentorAction(
                    label="Upload Dataset", type="add_node", payload={"node_type": "upload_file"}
                )
            )

        if "split" in gap_lower:
            actions.append(
                MentorAction(
                    label="Add Split Node", type="add_node", payload={"node_type": "split"}
                )
            )

        if "encoding" in gap_lower:
            actions.append(
                MentorAction(
                    label="Add Encoding", type="add_node", payload={"node_type": "encoding"}
                )
            )

        if "missing" in gap_lower:
            actions.append(
                MentorAction(
                    label="Add Missing Value Handler",
                    type="add_node",
                    payload={"node_type": "missing_value_handler"},
                )
            )

        return actions

    def _generate_next_step_actions(self, next_steps: List[str]) -> List[MentorAction]:
        """Generate actionable buttons from next steps."""
        actions = []

        for step in next_steps[:3]:  # Limit to 3 actions
            step_lower = step.lower()

            if "upload" in step_lower:
                actions.append(
                    MentorAction(
                        label="Upload Dataset",
                        type="add_node",
                        payload={"node_type": "upload_file"},
                    )
                )
            elif "split" in step_lower:
                actions.append(
                    MentorAction(label="Add Split", type="add_node", payload={"node_type": "split"})
                )
            elif "encoding" in step_lower:
                actions.append(
                    MentorAction(
                        label="Add Encoding", type="add_node", payload={"node_type": "encoding"}
                    )
                )
            elif "missing" in step_lower:
                actions.append(
                    MentorAction(
                        label="Handle Missing Values",
                        type="add_node",
                        payload={"node_type": "missing_value_handler"},
                    )
                )
            elif "execute" in step_lower:
                actions.append(MentorAction(label="Execute Pipeline", type="execute", payload={}))

        return actions

    def _personalize_message(
        self, base_message: str, personality: PersonalityStyle, expertise_level: ExpertiseLevel
    ) -> str:
        """Add personality and expertise-appropriate context to message."""

        # Add encouraging emoji/tone for beginners
        if expertise_level == ExpertiseLevel.BEGINNER:
            if personality == PersonalityStyle.ENCOURAGING:
                return base_message + "\n\n💡 You're doing great!"
            elif personality == PersonalityStyle.EDUCATIONAL:
                return base_message + "\n\n📚 Want to learn more about any of these steps?"

        return base_message

    def generate_model_introduction(
        self,
        model_type: str,
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
        expertise_level: ExpertiseLevel = ExpertiseLevel.BEGINNER,
    ) -> MentorSuggestion:
        """Generate detailed introduction and explanation for a specific model type."""

        if model_type == "linear_regression":
            intro = LINEAR_REGRESSION_GUIDE["introduction"]
            dataset_phase = LINEAR_REGRESSION_GUIDE["dataset_phase"]

            # Create the introduction message
            message = f"{intro['explanation']}\n\n{intro['visual_example']}\n\n---\n\n"
            message += f"**{dataset_phase['title']}**\n\n"
            message += "Before we start building, I need to know:"

            # Create actions for dataset options
            actions = []
            for question in dataset_phase["questions"]:
                for option in question["options"]:
                    actions.append(
                        MentorAction(
                            label=option["label"],
                            type=(
                                "show_guide"
                                if option["action"] == "request_upload"
                                else "learn_more"
                            ),
                            payload={
                                "action": option["action"],
                                "model_type": model_type,
                                "next_message": option["next_message"],
                            },
                        )
                    )

            return MentorSuggestion(
                id=str(uuid.uuid4()),
                type=SuggestionType.LEARNING_TIP,
                priority=SuggestionPriority.INFO,
                title=LINEAR_REGRESSION_GUIDE["title"],
                message=message,
                actions=actions,
                timestamp=datetime.utcnow().isoformat(),
                dismissible=False,
            )

        # Default fallback for other model types
        return MentorSuggestion(
            id=str(uuid.uuid4()),
            type=SuggestionType.LEARNING_TIP,
            priority=SuggestionPriority.INFO,
            title=f"{model_type.replace('_', ' ').title()} Guide",
            message="Let me help you build this model step by step!",
            actions=[
                MentorAction(
                    label="Start Building",
                    type="add_node",
                    payload={"node_type": "upload_file"},
                )
            ],
            timestamp=datetime.utcnow().isoformat(),
            dismissible=True,
        )

    def generate_dataset_guidance(
        self,
        action: str,
        model_type: str,
        next_message: str,
        personality: PersonalityStyle = PersonalityStyle.ENCOURAGING,
    ) -> MentorSuggestion:
        """Generate follow-up guidance after user selects a dataset option."""

        actions = []

        if action == "request_upload":
            # Guide user to upload dataset
            actions.append(
                MentorAction(
                    label="📂 Upload Dataset",
                    type="add_node",
                    payload={"node_type": "upload_file"},
                )
            )
        elif action == "provide_sample":
            # Offer sample datasets
            actions.extend(
                [
                    MentorAction(
                        label="🏠 Housing Dataset",
                        type="add_node",
                        payload={"node_type": "upload_file", "sample": "housing"},
                    ),
                    MentorAction(
                        label="📊 Student Performance",
                        type="add_node",
                        payload={"node_type": "upload_file", "sample": "students"},
                    ),
                    MentorAction(
                        label="📂 Upload My Own",
                        type="add_node",
                        payload={"node_type": "upload_file"},
                    ),
                ]
            )
        elif action == "explain_dataset":
            # After explanation, guide to upload
            actions.append(
                MentorAction(
                    label="Got it! Upload Dataset",
                    type="add_node",
                    payload={"node_type": "upload_file"},
                )
            )

        return MentorSuggestion(
            id=str(uuid.uuid4()),
            type=SuggestionType.NEXT_STEP,
            priority=SuggestionPriority.INFO,
            title="Next Step: Dataset",
            message=next_message,
            actions=actions,
            timestamp=datetime.utcnow().isoformat(),
            dismissible=True,
        )


# Singleton instance
recommendation_engine = RecommendationEngine()
