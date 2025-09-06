# =========================================================
# File: src/recommender/schemas.py
# Description:
#   Pydantic data models (schemas) for request and response validation
#   in the recommendation API.
#   Ensures strict input/output data structures for FastAPI endpoints.
# =========================================================

from pydantic import BaseModel, Field, ConfigDict
from typing import List

# =========================================================
# 1. Request Models
# =========================================================

# Request model for generating recommendations
class RecommendationRequest(BaseModel):
    user_id: str  # User identifier
    n_recommendations: int = 10  # Number of items to recommend
    recommendation_type: str = "hybrid"  # Algorithm type

# Request model for retrieving user history
class UserHistoryRequest(BaseModel):
    user_id: str  # User identifier
    limit: int = 50  # Max number of history items

# =========================================================
# 2. Response Models
# =========================================================

# Single recommended item schema
class Recommendation(BaseModel):
    item_id: str  # Unique item ID
    score: float  # Relevance score
    title: str  # Item title
    part: str  # Curriculum part name
    part_id: int  # Part identifier
    subjects: List[str]  # Associated subjects
    duration_minutes: float  # Estimated duration
    type: str  # Content type (e.g., lecture, quiz)

    # Example payload for documentation
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "item_id": "bundle_1",
                    "score": 0.85,
                    "title": "Intro to Grammar",
                    "part": "Part 1",
                    "part_id": 1,
                    "subjects": ["grammar", "english"],
                    "duration_minutes": 12.5,
                    "type": "lecture"
                }
            ]
        }
    )

# Response model for recommendation results
class RecommendationResponse(BaseModel):
    user_id: str  # User identifier
    recommendations: List[Recommendation]  # List of recommended items
    recommendation_type: str  # Algorithm type used

# Single user interaction history item
class HistoryItem(BaseModel):
    question_id: str  # Question attempted
    bundle_id: str  # Content bundle ID
    timestamp: str  # Interaction timestamp
    is_correct: bool  # Answer correctness
    elapsed_time: float  # Time spent
    part: str  # Curriculum part name
    subjects: List[str]  # Associated subjects

# Response model for user history
class UserHistoryResponse(BaseModel):
    user_id: str  # User identifier
    history: List[HistoryItem]  # List of past interactions