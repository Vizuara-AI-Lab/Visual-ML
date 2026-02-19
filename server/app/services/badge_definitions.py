"""
Badge and XP definitions for the gamification system.
All badge criteria and XP rewards are defined here.
"""

# XP thresholds for each level (cumulative)
# Level 1: 0 XP, Level 2: 100 XP, Level 3: 250 XP, etc.
LEVEL_THRESHOLDS = [
    0,      # Level 1
    100,    # Level 2
    250,    # Level 3
    450,    # Level 4
    700,    # Level 5
    1000,   # Level 6
    1400,   # Level 7
    1900,   # Level 8
    2500,   # Level 9
    3200,   # Level 10
    4000,   # Level 11
    5000,   # Level 12
    6200,   # Level 13
    7600,   # Level 14
    9200,   # Level 15
    11000,  # Level 16
    13000,  # Level 17
    15500,  # Level 18
    18500,  # Level 19
    22000,  # Level 20
]

# XP awards for different actions
XP_AWARDS = {
    "pipeline_execution": 25,
    "project_creation": 15,
    "first_login": 50,
    "quiz_passed": 20,
    "activity_completed": 30,
    "story_completed": 40,
    "app_published": 35,
    "dataset_uploaded": 10,
    "project_shared": 15,
}

# Badge definitions — keys must be unique
BADGES = {
    "welcome": {
        "name": "Welcome!",
        "description": "Logged in for the first time",
        "icon": "rocket",
        "color": "#3B82F6",
    },
    "first_pipeline": {
        "name": "First Pipeline",
        "description": "Successfully executed your first ML pipeline",
        "icon": "zap",
        "color": "#10B981",
    },
    "data_explorer": {
        "name": "Data Explorer",
        "description": "Used 5 different view nodes to explore data",
        "icon": "search",
        "color": "#06B6D4",
    },
    "model_master": {
        "name": "Model Master",
        "description": "Trained 10 different ML models",
        "icon": "brain",
        "color": "#8B5CF6",
    },
    "perfect_score": {
        "name": "Perfect Score",
        "description": "Achieved R2 score above 0.95 on a model",
        "icon": "trophy",
        "color": "#F59E0B",
    },
    "quiz_whiz": {
        "name": "Quiz Whiz",
        "description": "Scored 100% on 3 quizzes",
        "icon": "star",
        "color": "#EF4444",
    },
    "clean_machine": {
        "name": "Clean Machine",
        "description": "Used missing value handler, encoding, and scaling in one pipeline",
        "icon": "sparkles",
        "color": "#A855F7",
    },
    "app_creator": {
        "name": "App Creator",
        "description": "Published your first custom app",
        "icon": "layout",
        "color": "#EC4899",
    },
    "activity_explorer": {
        "name": "Activity Explorer",
        "description": "Completed all 5 interactive activities",
        "icon": "gamepad",
        "color": "#14B8A6",
    },
    "story_complete": {
        "name": "Storyteller",
        "description": "Completed your first dataset story",
        "icon": "book-open",
        "color": "#F97316",
    },
}
